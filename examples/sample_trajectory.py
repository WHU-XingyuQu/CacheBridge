"""Runnable CPU contract example using a toy environment and backend.

This is an integration example, not an agent-quality or GPU benchmark.
Replace the environment/backend with real implementations of the same methods.
"""

from cachebridge.trajectory import (
    Generation,
    Message,
    Observation,
    RouteDecision,
    TaskRef,
)
from cachebridge.agents import CacheBridgePolicy, sample_trajectory
from cachebridge.offline.trajectory import handoff_samples


class ExampleEnvironment:
    def reset(self, task, *, seed):
        self.messages = (Message("user", "Look up the order and resolve the request."),)
        self.calls = 0
        return Observation(self.messages)

    def step(self, response):
        self.calls += 1
        self.messages += (response,)
        if self.calls == 3:
            return Observation(
                self.messages, done=True, stop_reason="environment_terminal"
            )
        self.messages += (Message("tool", "Lookup completed.", name="lookup_order"),)
        return Observation(self.messages)


class ExampleRouter:
    def choose(self, context):
        ready = any(
            m.role == "tool" and m.name == "lookup_order" for m in context.messages
        )
        return RouteDecision(
            "large" if ready else "small", "resolve_after_lookup" if ready else "intake"
        )


class ExampleBackend:
    def begin_episode(self, trajectory_id):
        self.calls = 0

    def generate(self, request):
        prompts = ((1, 2), (1, 2, 3, 4), (1, 2, 3, 4, 5, 6, 7))
        outputs = ((3,), (5, 6), (8, 9))
        step = self.calls
        self.calls += 1
        mode = request.cache_plan.mode
        return Generation(
            response=Message("assistant", ("look up", "prepare", "resolve")[step]),
            prompt_ids=prompts[step],
            completion_ids=outputs[step],
            tokenizer_id="toy-tokenizer-v1",
            model_id=request.model_id,
            executed_mode=mode,
            reused_tokens=3 if mode == "cachebridge" else (6 if mode == "reuse" else 0),
            handoff_prefix_ids=(1, 2, 3) if step == 1 else (),
            mapper_id=request.cache_plan.mapper_id if mode == "cachebridge" else None,
        )

    def end_episode(self, trajectory_id):
        pass


def main():
    task = TaskRef("example", "orders", "train", "0")
    for enabled in (False, True):
        trajectory = sample_trajectory(
            task,
            run_id="example/mapped" if enabled else "example/native",
            environment=ExampleEnvironment(),
            router=ExampleRouter(),
            backend=ExampleBackend(),
            models=("small", "large"),
            cache_policy=CacheBridgePolicy(
                {("small", "large"): "local-mapper-v1"}, enabled=enabled
            ),
            seed=7,
        )
        assert trajectory.status == "terminal", trajectory.error
        print(
            "CacheBridge:",
            enabled,
            "models:",
            [t.route.model_id for t in trajectory.turns],
            "execution:",
            [t.generation.executed_mode for t in trajectory.turns],
        )
        if not enabled:
            samples = handoff_samples(trajectory)
            print("Native TRAIN sample:", samples[0].to_dict())


if __name__ == "__main__":
    main()
