import concurrent.futures
import dataclasses
import functools
import pathlib
import pickle
from typing import Sequence

import einops
from flax import struct
import flax.nnx as nnx
import imageio
import jax
import jax.numpy as jnp
import kinetix.environment.env as kenv
import kinetix.environment.env_state as kenv_state
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import tyro
import wandb

import eval_flow as _eval
import generate_data
import model as _model
import train_expert

WANDB_PROJECT = "rtc-kinetix-residual"
LOG_DIR = pathlib.Path("logs-residual")


@dataclasses.dataclass(frozen=True)
class Config:
    run_path: str
    base_policy_path: str  # 事前学習されたFlowPolicyのパス（残差学習のベースとなる）
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
    )
    batch_size: int = 512
    num_epochs: int = 16
    seed: int = 0

    # Residual Policy specific configs
    residual_learning_rate: float = 1e-4
    residual_weight_decay: float = 1e-3
    grad_norm_clip: float = 5.0
    lr_warmup_steps: int = 500

    eval: _eval.EvalConfig = _eval.EvalConfig()


@struct.dataclass
class ResidualEpochCarry:
    rng: jax.Array
    train_state: nnx.State
    graphdef: nnx.GraphDef[tuple[_model.ResidualPolicy, nnx.Optimizer]]


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
    
    # Freeze base policy
    frozen_params = jax.tree_map(jax.lax.stop_gradient, nnx.state(policy, nnx.Param))
    nnx.update(policy, frozen_params)
    return policy


def create_residual_batch(
    rng: jax.Array,
    base_policy: _model.FlowPolicy,
    obs_chunks: jax.Array,
    action_chunks: jax.Array,
    action_chunk_size: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Create training batch for residual policy learning.
    
    Residual learning concept:
    - Base policy generates base_action from observation
    - Target is the expert/optimal action
    - Residual = target_action - base_action (what the residual policy should learn)
    - Final action = base_action + residual
    
    Args:
        obs_chunks: [batch_size, chunk_size, obs_dim] - observation sequence
        action_chunks: [batch_size, action_chunk_size, action_dim] - target actions from expert
    
    Returns:
        batch_obs: [batch_size, obs_dim] - observations used by residual policy 
        batch_base_actions: [batch_size, action_dim] - base policy actions
        batch_target_actions: [batch_size, action_dim] - target expert actions
    """
    batch_size = obs_chunks.shape[0]
    
    first_obs = obs_chunks[:, 0, :]  # [batch_size, obs_dim]
    
    # Generate base policy action using first obs (single action, not chunk)
    rng_split = jax.random.split(rng, batch_size + 1)[1:]  # Split RNG for each  batch item
    base_actions = jax.vmap(
        lambda o, rng: base_policy.action(rng, o[None],num_steps=5),
        in_axes=(0, 0)
    )(first_obs, rng_split)
    
    # select time t from 0 to chunk_size-1 for residual learning
    # This is the time step at which we will extract observations and actions
    # We use a random time t for each batch item to allow diverse training
    # This allows the residual policy to learn corrections at different time steps
    time_t = jax.random.randint(rng, (batch_size,), 0, action_chunk_size)
    #select base actions at time t
    base_actions = base_actions.squeeze(axis=1) # [batch_size,action_chunk_size, action_dim]
    base_actions = base_actions[jnp.arange(batch_size), time_t, :]  # [ 
    base_actions = jax.lax.stop_gradient(base_actions)
    
    # Extract observations at time t
    batch_obs = obs_chunks[jnp.arange(batch_size), time_t, :]  # [batch_size, obs_dim]
    # Extract target actions at time t
    batch_target_actions = action_chunks[jnp.arange(batch_size), time_t, :]  # [batch_size, action_dim]
    
    # Change time t to cos(t) for residual learning
    time_features_cos = jnp.cos(time_t * (2 * jnp.pi / action_chunk_size))  # [batch_size]
    time_features_sin = jnp.sin(time_t * (2 * jnp.pi / action_chunk_size))  # [batch_size]
    time_features = jnp.stack([time_features_cos, time_features_sin], axis=-1)  # [batch_size, 2]
    
    return batch_obs, base_actions, time_features, batch_target_actions


def main(config: Config):
    static_env_params = kenv_state.StaticEnvParams(**train_expert.LARGE_ENV_PARAMS, frame_skip=train_expert.FRAME_SKIP)
    env_params = kenv_state.EnvParams()
    levels = train_expert.load_levels(config.level_paths, static_env_params, env_params)
    static_env_params = static_env_params.replace(screen_dim=train_expert.SCREEN_DIM)

    env = kenv.make_kinetix_env_from_name("Kinetix-Symbolic-Continuous-v1", static_env_params=static_env_params)

    mesh = jax.make_mesh((jax.local_device_count(),), ("level",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("level"))

    action_chunk_size = config.eval.model.action_chunk_size

    # Load data
    def load_data(level_path: str):
        level_name = level_path.replace("/", "_").replace(".json", "")
        return dict(np.load(pathlib.Path(config.run_path) / "data" / f"{level_name}.npz"))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        data = list(executor.map(load_data, config.level_paths))
    
    with jax.default_device(jax.devices("cpu")[0]):
        data = jax.tree.map(lambda *x: einops.rearrange(jnp.stack(x), "l s e ... -> l (e s) ..."), *data)
        valid_steps = data["obs"].shape[1] - action_chunk_size + 1
        data = jax.tree.map(
            lambda x: x[:, : (valid_steps // config.batch_size) * config.batch_size + action_chunk_size - 1], data
        )
        data = jax.tree.map(
            lambda x: jax.make_array_from_single_device_arrays(
                x.shape,
                sharding,
                [
                    jax.device_put(y, d)
                    for y, d in zip(jnp.split(x, jax.local_device_count()), jax.local_devices(), strict=True)
                ],
            ),
            data,
        )

    data: generate_data.Data = generate_data.Data(**data)
    print(f"Truncated data to {data.obs.shape[1]:_} steps ({valid_steps // config.batch_size:_} batches)")

    obs_dim = data.obs.shape[-1]
    action_dim = env.action_space(env_params).shape[0]

    # Load base policies for each level
    def load_base_policy_for_level(level_idx: int):
        level_name = config.level_paths[level_idx].replace("/", "_").replace(".json", "")
        policy_path = pathlib.Path(config.base_policy_path) / f"{level_name}.pkl"
        return load_base_policy(str(policy_path), obs_dim, action_dim, config.eval.model)

    base_policies = [load_base_policy_for_level(i) for i in range(len(config.level_paths))]
    base_policy = jax.tree.map(lambda *x: x[0], *base_policies)  # Use the first policy as base for vmap

    @functools.partial(jax.jit, in_shardings=sharding, out_shardings=sharding)
    @jax.vmap
    def init(rng: jax.Array) -> ResidualEpochCarry:
        rng, key = jax.random.split(rng)
        
        residual_config = _model.ResidualModelConfig(
            channel_dim=256,
            channel_hidden_dim=512,
        )
        
        residual_policy = _model.ResidualPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            config=residual_config,
            rngs=nnx.Rngs(key),
        )
        
        total_params = sum(x.size for x in jax.tree.leaves(nnx.state(residual_policy, nnx.Param)))
        print(f"Residual policy total params: {total_params:,}")
        
        optimizer = nnx.Optimizer(
            residual_policy,
            optax.chain(
                optax.clip_by_global_norm(config.grad_norm_clip),
                optax.adamw(
                    optax.warmup_constant_schedule(0, config.residual_learning_rate, config.lr_warmup_steps),
                    weight_decay=config.residual_weight_decay,
                ),
            ),
        )
        
        graphdef, train_state = nnx.split((residual_policy, optimizer))
        return ResidualEpochCarry(rng, train_state, graphdef)


    def make_train_epoch(
        base_policy: _model.FlowPolicy,
    ):
        @functools.partial(jax.jit, donate_argnums=(0,), in_shardings=sharding, out_shardings=sharding)
        @jax.vmap
        def train_epoch(epoch_carry: ResidualEpochCarry, level: kenv_state.EnvState, data: generate_data.Data):
            def train_minibatch(carry: tuple[jax.Array, nnx.State], batch_idxs: jax.Array):
                rng, train_state = carry
                residual_policy, optimizer = nnx.merge(epoch_carry.graphdef, train_state)

                rng, key = jax.random.split(rng)

                def loss_fn(residual_policy: _model.ResidualPolicy):
                    # Create observation chunks [batch_size, chunk_size, obs_dim]
                    # Safely get obs[t] to obs[t+chunk_size-1] 
                    obs_chunks = data.obs[batch_idxs[:, None] + jnp.arange(action_chunk_size)[None, :]]
                    action_chunks = data.action[batch_idxs[:, None] + jnp.arange(action_chunk_size)[None, :]]
                    
                    # Zero actions after done
                    done_chunks = data.done[batch_idxs[:, None] + jnp.arange(action_chunk_size
                    )[None, :]]
                    done_idxs = jnp.where(
                        jnp.any(done_chunks, axis=-1),
                        jnp.argmax(done_chunks, axis=-1),
                        action_chunk_size,
                    )
                    action_chunks = jnp.where(
                        jnp.arange(action_chunk_size)[None, :, None] >= done_idxs[:, None, None],
                        0.0,
                        action_chunks,
                    )   
                    # Create residual training batch with observation chunks
                    batch_obs, batch_base_actions, batch_time_features, batch_target_actions = create_residual_batch(
                        key, base_policy, obs_chunks, action_chunks, action_chunk_size
                    )   
                    # Residual policy learns: residual = target_action - base_action
                    # This allows the policy to focus on corrections rather than full action prediction
                    return residual_policy.loss(batch_obs, batch_base_actions, batch_time_features, batch_target_actions)
                loss, grads = nnx.value_and_grad(loss_fn)(residual_policy)
                info = {"loss": loss, "grad_norm": optax.global_norm(grads)}
                optimizer.update(grads)
                _, train_state = nnx.split((residual_policy, optimizer))
                return (rng, train_state), info
            # Shuffle and batch (ensure we have space for observation sequences)
            rng, key = jax.random.split(epoch_carry.rng)
            permutation = jax.random.permutation(key, data.obs.shape[0] - action_chunk_size + 1)
            
            # Batch
            permutation = permutation.reshape(-1, config.batch_size)
            # Train
            (rng, train_state), train_info = jax.lax.scan(
                train_minibatch, (epoch_carry.rng, epoch_carry.train_state), permutation
            )
            train_info = jax.tree.map(lambda x: x.mean(), train_info)
            
            eval_info = {}
            video = None
            return ResidualEpochCarry(rng, train_state, epoch_carry.graphdef), ({**train_info, **eval_info}, video)
        return train_epoch
            
    wandb.init(project=WANDB_PROJECT)
    rng = jax.random.key(config.seed)
    
    # Initialize residual policies  
    epoch_carry = init(
        jax.random.split(rng, len(config.level_paths)), 
    )
    train_epoch = make_train_epoch(base_policy)
    for epoch_idx in tqdm.tqdm(range(config.num_epochs)):
        epoch_carry, (info, video) = train_epoch(epoch_carry, levels, data)

        for i in range(len(config.level_paths)):
            level_name = config.level_paths[i].replace("/", "_").replace(".json", "")
            wandb.log({f"residual_{level_name}/{k}": v[i] for k, v in info.items()}, step=epoch_idx)

            log_dir = LOG_DIR / wandb.run.name / str(epoch_idx)

            if video is not None:
                video_dir = log_dir / "videos"
                video_dir.mkdir(parents=True, exist_ok=True)
                imageio.mimwrite(video_dir / f"residual_{level_name}.mp4", video[i], fps=15)

            # Save residual policy
            policy_dir = log_dir / "residual_policies"
            policy_dir.mkdir(parents=True, exist_ok=True)
            level_train_state = jax.tree.map(lambda x: x[i], epoch_carry.train_state)
            with (policy_dir / f"residual_{level_name}.pkl").open("wb") as f:
                residual_policy, _ = nnx.merge(epoch_carry.graphdef, level_train_state)
                state_dict = nnx.state(residual_policy).to_pure_dict()
                pickle.dump(state_dict, f)


if __name__ == "__main__":
    tyro.cli(main)
