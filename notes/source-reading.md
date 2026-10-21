# Source reading: observation, sampling, and execution

These notes summarize existing project notes after checking the locally installed
LeRobot 0.3.3 and gym-pusht 0.1.5 source. The four files under the original
`source/` directory matched their installed counterparts byte for byte. Those
upstream files are not vendored here. Function locations below refer to the
installed distribution, not necessarily to the current upstream branch.

1. `lerobot.envs.utils.preprocess_observation`: changes `pixels` (uint8,
   channel-last) to `observation.image` (float32, channel-first, divided by 255),
   and `agent_pos` to `observation.state`. Captured shapes are `[1,3,96,96]`
   and `[1,2]`. The evaluator transfers tensors to CUDA.
2. `DiffusionPolicy.select_action`: normalizes observations and updates a
   two-observation history on every control step. Initially the first observation
   fills the history. Images use MEAN_STD normalization; state uses MIN_MAX.
   Evaluation uses an 84×84 center crop and ResNet18 visual encoding.
3. `DiffusionModel.conditional_sample`: samples Gaussian noise of shape
   `[B,16,2]`; a conditional 1D U-Net and DDPM scheduler iteratively denoise it.
   `num_inference_steps=None` resolves to the configured 100 training timesteps.
   These 100 denoising iterations are distinct from environment control steps.
4. `DiffusionModel.generate_actions`: selects actions beginning at
   `n_obs_steps - 1` from the generated horizon. `select_action` unnormalizes
   them and queues 4 or 8 actions. It returns one `[B,2]` action per call;
   new diffusion sampling happens when the queue is empty. Thus new observations
   do not change actions already in the queue.
5. `lerobot.scripts.eval.rollout`: moves the action to CPU/NumPy and calls
   `env.step`. The action is a target pusher `[x,y]` position in the environment's
   512×512 coordinate space, not an observation-image pixel or joint angle.
6. `gym_pusht.envs.pusht.PushTEnv.step`: uses a PD controller and Pymunk
   integration. Success requires target coverage strictly greater than 0.95;
   reward is `clip(coverage / 0.95, 0, 1)`. The configured time limit is 300
   control steps at 10 Hz. Physics and rendering run on CPU.

The first-step trace records input/output shapes and device placement. It does
not directly record intermediate normalization, visual features, or each DDPM
iteration; those details come from source inspection and configuration.

The action logger's `replanned` flag is inferred from the action queue being
empty before `select_action`, rather than from a direct sampler profiler.
It is sufficient to inspect the queue pattern for this version of LeRobot.
Only the pusher state and target action are logged; these traces cannot establish
the T object's full pose, contact forces, or a causal explanation for failure.

References: [LeRobot v0.3.3](https://github.com/huggingface/lerobot/tree/v0.3.3),
[gym-pusht](https://github.com/huggingface/gym-pusht),
[Diffusion Policy](https://diffusion-policy.cs.columbia.edu/).
