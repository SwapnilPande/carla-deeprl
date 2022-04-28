from projects.morel_mopo.algorithm.fake_envs import fake_env_utils as feutils
import torch

class PredictionWrapper():
    """
    Wrapper class for the dynamics model.

    This class wraps the dynamics model to handle maintaining the state
    for the dynamics input. Namely, this will handle maintaining the history
    required for the frame stack and provide the inputs in the right order
    for the particular model.
    """

    def __init__(self, dynamics, delta_update_func, current_idx):
        self.dynamics = dynamics

        # Retrieve important parameters from the dynamics model
        self.frame_stack = self.dynamics.frame_stack
        self.normalization_stats = self.dynamics.normalization_stats
        self.device = self.dynamics.device

        # Create the state buffer
        self.states = feutils.NormalizedTensor(
            self.normalization_stats["obs"]["mean"],
            self.normalization_stats["obs"]["std"],
            self.device
        )
        # Create the action buffer
        self.actions = feutils.NormalizedTensor(
            self.normalization_stats["action"]["mean"],
            self.normalization_stats["action"]["std"],
            self.device
        )

        # Buffer to save output of dynamics model
        self.deltas = feutils.NormalizedTensor(
            self.normalization_stats["delta"]["mean"],
            self.normalization_stats["delta"]["std"],
            self.device
        )

        self.delta_update_func = delta_update_func

        self.current_idx = current_idx

    def reset(self, initial_states: torch.Tensor, initial_actions: torch.Tensor):
        """ Reset the state of the dynamics model.

        Args:
            initial_states (torch.Tensor): Initial states of the agents [n_agents, frame_stack, state_dim]
            initial_actions (torch.Tensor): Initial actions of the agents [n_agents, frame_stack, action_dim]
        """
        # Make sure that the input is in the right shape
        assert (initial_states.shape[-2] == self.frame_stack and
                initial_states.shape[-1] == self.dynamics.state_dim_in)

        assert (initial_actions.shape[-2] == self.frame_stack and
                initial_actions.shape[-1] == self.dynamics.action_dim)

        # Save states in NormalizedTensors
        self.states.unnormalized = initial_states
        self.actions.unnormalized = initial_actions

        # Sample a model_idx for this rollout
        self.model_idx = torch.randint(0, self.dynamics.n_models, (1,)).item()

        return self.states.unnormalized[:, self.current_idx, :], self.actions.unnormalized[:, self.current_idx, :]


    def update_state(self, delta_state: torch.Tensor):
        """ Update the state of the dynamics model.

        Args:
            delta_state (torch.Tensor): Delta state of the agents [n_agents, state_dim]
        """

        raise NotImplementedError

    def update_action(self, new_action: torch.Tensor):
        """ Update the action of the dynamics model.

        Args:
            new_action (torch.Tensor): New action of the agents [n_agents, action_dim]
        """

        raise NotImplementedError

    def make_prediction(self) -> torch.Tensor:
        """ Make a forward pass to compute deltas

        Returns:
            next_state (torch.Tensor): Next state of the agents [n_agents, frame_stack, state_dim]
        """

        # Get predictions across all models
        try:
            deltas, rewards = self.dynamics.predict(self.states.normalized, self.actions.normalized, model_idx = self.model_idx)
            return deltas
        except:
            import ipdb; ipdb.set_trace()

            return torch.zeros()


    def step(self, action: torch.Tensor):
        """ Step the dynamics model forward.

        Args:
            action (torch.Tensor): Action of the agents [n_agents, action_dim]

        Returns:
            next_state (torch.Tensor): Next state of the agents [n_agents, state_dim]
            additional_output: Additional output from the state_update_function
        """
        with torch.no_grad():

            # Update the action buffer
            self.update_action(action)

            # Run forward pass of model
            self.deltas.normalized = torch.clone(
                self.make_prediction()
            )

            # Compute new state from deltas
            state_delta, additional_output = self.delta_update_func(self.deltas)

            # Update the state buffer
            self.update_state(state_delta)

            # Return the new state
            return self.states.unnormalized[:, self.current_idx, :], additional_output




