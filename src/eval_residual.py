import collections
import dataclasses
import functools
import math
import pathlib
import pickle
from typing import Sequence

import flax.nnx as nnx
import jax
from jax.experimental import shard_map
import jax.numpy as jnp
import kinetix.environment.env as kenv
import kinetix.environment.env_state as kenv_state
import kinetix.environment.wrappers as wrappers
import kinetix.render.renderer_pixels as renderer_pixels
import pandas as pd
import tyro

import model as _model
import train_expert


@dataclasses.dataclass(frozen=True)
class ResidualEvalConfig:
    """Configuration for evaluating residual policies."""
    step: int = -1  # Which epoch to evaluate (-1 for latest)
    num_evals: int = 2048  # Number of evaluation episodes
    num_flow_steps: int = 5

    inference_delay: int = 0
    execute_horizon: int = 1
    # Residual policy specific
    base_policy_path: str | None = None  # Path to base policy if different from training
    model: _model.ModelConfig = _model.ModelConfig()
    residual_model: _model.ResidualModelConfig = _model.ResidualModelConfig()


def load_base_policy(policy_path: str, obs_dim: int, action_dim: int, config: _model.ModelConfig) -> _model.FlowPolicy:
    """Load pre-trained FlowPolicy from pickle file."""
    with open(policy_path, "rb") as f:
        state_dict = pickle.load(f)
    
    # Create policy with dummy RNG
    dummy_rng = jax.random.PRNGKey(0)
    policy = _model.FlowPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        config=config,
        rngs=nnx.Rngs(dummy_rng),
    )
    
    # Load state
    policy_state = nnx.state(policy)
    policy_state.update(state_dict)
    nnx.update(policy, policy_state)
    
    return policy


def load_residual_policy(
    policy_path: str, 
    obs_dim: int, 
    action_dim: int, 
    config: _model.ResidualModelConfig
) -> _model.ResidualPolicy:
    """Load pre-trained ResidualPolicy from pickle file."""
    with open(policy_path, "rb") as f:
        state_dict = pickle.load(f)
    
    # Create residual policy with dummy RNG
    dummy_rng = jax.random.PRNGKey(0)
    policy = _model.ResidualPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        config=config,
        rngs=nnx.Rngs(dummy_rng),
    )
    
    # Load state
    policy_state = nnx.state(policy)
    policy_state.update(state_dict)
    nnx.update(policy, policy_state)
    
    return policy


def eval(
    config: ResidualEvalConfig,
    env: kenv.environment.Environment,
    rng: jax.Array,
    level: kenv_state.EnvState,
    base_policy: _model.FlowPolicy,
    residual_policy: _model.ResidualPolicy,
    env_params: kenv_state.EnvParams,
    static_env_params: kenv_state.EnvParams,
):
    env = train_expert.BatchEnvWrapper(
        wrappers.LogWrapper(wrappers.AutoReplayWrapper(train_expert.NoisyActionWrapper(env))), config.num_evals
    )
    render_video = train_expert.make_render_video(renderer_pixels.make_render_pixels(env_params, static_env_params))
    assert config.execute_horizon >= config.inference_delay, f"{config.execute_horizon=} {config.inference_delay=}"

    def execute_chunk(carry, _):
        def step_with_residual_policy(carry, action):
            rng, obs, env_state,time  = carry
            if time < config.inference_delay:
                actual_time = time + config.execute_horizon
            else:
                actual_time = time
            rng, key = jax.random.split(rng)
            time_feature = jnp.cos(actual_time * (2 * jnp.pi / base_policy.action_chunk_size))
            final_action = residual_policy.apply_residual(obs=obs,base_action=action,time_feature=time_feature)
            next_obs, next_env_state, reward, done, info = env.step(key, env_state, final_action, env_params)
            time += 1
            return (rng, next_obs, next_env_state,time), (done, env_state, info)

        rng, obs, env_state, action_chunk, n = carry
        rng, key = jax.random.split(rng)
        next_action_chunk = base_policy.action(key, obs, config.num_flow_steps)

        # we execute `inference_delay` actions from the *previously generated* action chunk, and then the remaining
        # `execute_horizon - inference_delay` actions from the newly generated action chunk
        action_chunk_to_execute = jnp.concatenate(
            [
                action_chunk[:, : config.inference_delay],
                next_action_chunk[:, config.inference_delay : config.execute_horizon],
            ],
            axis=1,
        )
        # throw away the first `execute_horizon` actions from the newly generated action chunk, to align it with the
        # correct frame of reference for the next scan iteration
        next_action_chunk = jnp.concatenate(
            [
                next_action_chunk[:, config.execute_horizon :],
                jnp.zeros((obs.shape[0], config.execute_horizon, base_policy.action_dim)),
            ],
            axis=1,
        )
        next_n = jnp.concatenate([n[config.execute_horizon :], jnp.zeros(config.execute_horizon, dtype=jnp.int32)])
        time = 0
        (rng, next_obs, next_env_state), (dones, env_states, infos) = jax.lax.scan(
            step_with_residual_policy, (rng, obs, env_state,time), action_chunk_to_execute.transpose(1, 0, 2)
        )
        # if config.inference_delay > 0:
        #     infos["match"] = jnp.mean(jnp.abs(fixed_prefix - action_chunk_to_execute))
        return (rng, next_obs, next_env_state, next_action_chunk, next_n), (dones, env_states, infos)

    rng, key = jax.random.split(rng)
    obs, env_state = env.reset_to_level(key, level, env_params)
    rng, key = jax.random.split(rng)
    action_chunk = base_policy.action(key, obs, config.num_flow_steps)  # [batch, horizon, action_dim]
    n = jnp.ones(action_chunk.shape[1], dtype=jnp.int32)
    scan_length = math.ceil(env_params.max_timesteps / config.execute_horizon)
    _, (dones, env_states, infos) = jax.lax.scan(
        execute_chunk,
        (rng, obs, env_state, action_chunk, n),
        None,
        length=scan_length,
    )
    dones, env_states, infos = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), (dones, env_states, infos))
    assert dones.shape[0] >= env_params.max_timesteps, f"{dones.shape=}"
    return_info = {}
    for key in ["returned_episode_returns", "returned_episode_lengths", "returned_episode_solved"]:
        # only consider the first episode of each rollout
        first_done_idx = jnp.argmax(dones, axis=0)
        return_info[key] = infos[key][first_done_idx, jnp.arange(config.num_evals)].mean()
    for key in ["match"]:
        if key in infos:
            return_info[key] = jnp.mean(infos[key])
    video = render_video(jax.tree.map(lambda x: x[:, 0], env_states))
    return return_info, video

def main(
    run_path: str,
    config: ResidualEvalConfig = ResidualEvalConfig(),
    level_paths: Sequence[str] = (
        "worlds/l/grasp_easy.json",
        #"worlds/l/catapult.json", 
        #"worlds/l/cartpole_thrust.json",
        #"worlds/l/hard_lunar_lander.json",
        #"worlds/l/mjc_half_cheetah.json",
        #"worlds/l/mjc_swimmer.json",
        #"worlds/l/mjc_walker.json",
        #"worlds/l/h17_unicycle.json",
        #"worlds/l/chain_lander.json",
        #"worlds/l/catcher_v3.json",
        #"worlds/l/trampoline.json",
        #"worlds/l/car_launch.json",
    ),
    seed: int = 0,
    output_dir: str | None = "eval_residual_output",
):
    """
    Evaluate residual policies and compare with base policies.
    
    Args:
        run_path: Path to the training run directory containing residual policies
        config: Evaluation configuration
        level_paths: List of level files to evaluate on
        seed: Random seed
        output_dir: Directory to save evaluation results
    """
    print(f"Evaluating residual policies from: {run_path}")
    
    # Setup environment
    static_env_params = kenv_state.StaticEnvParams(**train_expert.LARGE_ENV_PARAMS, frame_skip=train_expert.FRAME_SKIP)
    env_params = kenv_state.EnvParams()
    levels = train_expert.load_levels(level_paths, static_env_params, env_params)
    static_env_params = static_env_params.replace(screen_dim=train_expert.SCREEN_DIM)

    env = kenv.make_kinetix_env_from_name("Kinetix-Symbolic-Continuous-v1", static_env_params=static_env_params)
    
    # Get dimensions
    obs_dim = jax.eval_shape(env.reset_to_level, jax.random.key(0), jax.tree.map(lambda x: x[0], levels), env_params)[0].shape[-1]
    action_dim = env.action_space(env_params).shape[0]

    # Find the run directory
    run_path = pathlib.Path(run_path)
    log_dirs = list(filter(lambda p: p.is_dir() and p.name.isdigit(), run_path.iterdir()))
    log_dirs = sorted(log_dirs, key=lambda p: int(p.name))
    
    if config.step == -1:
        eval_dir = log_dirs[-1]  # Latest epoch
    else:
        eval_dir = log_dirs[config.step]
    
    print(f"Evaluating epoch: {eval_dir.name}")

    # Setup JAX devices
    mesh = jax.make_mesh((jax.local_device_count(),), ("x",))
    pspec = jax.sharding.PartitionSpec("x")
    sharding = jax.sharding.NamedSharding(mesh, pspec)

    rngs = jax.random.split(jax.random.key(seed), len(level_paths))
    results = collections.defaultdict(list)
    
    print("=" * 60)
    print("RESIDUAL POLICY EVALUATION RESULTS")
    print("=" * 60)
    
    for i, level_path in enumerate(level_paths):
        level_name = level_path.replace("/", "_").replace(".json", "")
        print(f"Evaluating level: {level_name}")
        
        # Load base policy
        if config.base_policy_path:
            base_policy_path = pathlib.Path(config.base_policy_path) / f"{level_name}.pkl"
        else:
            # Assume base policy is in the same structure
            base_policy_path = eval_dir / "base_policies" / f"{level_name}.pkl"
            if not base_policy_path.exists():
                raise FileNotFoundError(f"Base policy not found at {base_policy_path}, please provide a valid path.")
        
        base_policy = load_base_policy(str(base_policy_path), obs_dim, action_dim, config.model)
        
        # Load residual policy
        residual_policy_path = eval_dir / "residual_policies" / f"residual_{level_name}.pkl"
        if not residual_policy_path.exists():
            raise FileNotFoundError(f"Residual policy not found at {residual_policy_path}, please provide a valid path.")
            
        residual_policy = load_residual_policy(
            str(residual_policy_path), obs_dim, action_dim, config.residual_model
        )
        
        
        mesh = jax.make_mesh((jax.local_device_count(),), ("x",))
    pspec = jax.sharding.PartitionSpec("x")
    sharding = jax.sharding.NamedSharding(mesh, pspec)

    @functools.partial(jax.jit, static_argnums=(0,), in_shardings=sharding, out_shardings=sharding)
    @functools.partial(shard_map.shard_map, mesh=mesh, in_specs=(None, pspec, pspec), out_specs=pspec)
    @functools.partial(jax.vmap, in_axes=(None, 0, 0,))
    def _eval(config: ResidualEvalConfig, rng: jax.Array, level: kenv_state.EnvState):
        eval_info, _ = eval(config, env, rng, level, base_policy,residual_policy, env_params, static_env_params)
        return eval_info

    rngs = jax.random.split(jax.random.key(seed), len(level_paths))
    results = collections.defaultdict(list)
    for inference_delay in [0, 1, 2, 3, 4]:
        for execute_horizon in range(max(1, inference_delay), 8 - inference_delay + 1):
            print(f"{inference_delay=} {execute_horizon=}")
            c = dataclasses.replace(
                config, inference_delay=inference_delay, execute_horizon=execute_horizon
            )
            out = jax.device_get(_eval(c, rngs, levels))
            for i in range(len(level_paths)):
                for k, v in out.items():
                    results[k].append(v[i])
                results["delay"].append(inference_delay)
                results["method"].append("naive")
                results["level"].append(level_paths[i])
                results["execute_horizon"].append(execute_horizon)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(pathlib.Path(output_dir) / "results.csv", index=False)

if __name__ == "__main__":
    tyro.cli(main)
