from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
import numpy as np

class TensorboardLogger(BaseCallback):
    """
     A custom callback for tensorboard logging.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.n_episodes = 0
        self.ep_reward = 0
        self.ep_length = 0

    # Those variables will be accessible in the callback
    # (they are defined in the base class)
    # The RL model
    # self.model = None  # type: BaseAlgorithm
    # An alias for self.model.get_env(), the environment used for training
    # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
    # Number of time the callback was called
    # self.n_calls = 0  # type: int
    # self.num_timesteps = 0  # type: int
    # local and global variables
    # self.locals = None  # type: Dict[str, Any]
    # self.globals = None  # type: Dict[str, Any]
    # The logger object, used to report things in the terminal
    # self.logger = None  # stable_baselines3.common.logger
    # # Sometimes, for event callback, it is useful
    # # to have access to the parent object
    # self.parent = None  # type: Optional[BaseCallback]

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """

        # Logging data at the end of an episode - must check if the environment is done
        done_array = self.locals["dones"]
        n_done = np.sum(done_array).item()

        # Only log if any workers are actually at the end of an episode
        if n_done > 0:
            # Record the cumulative number of finished episodes
            self.n_episodes += n_done
            self.logger.record('time/episodes', self.n_episodes)

            # Fetch data from the info dictionary in the environment (convert tuple->np.ndarray for easy indexing)
            infos = np.array(self.locals["infos"])[done_array]

            avg_reward = 0
            avg_length = 0
            avg_collision_reward = 0
            avg_collision_avoidance_reward = 0
            avg_pf_reward = 0
            avg_close_to_target_reward = 0
            for info in infos:
                avg_reward += info["reward"]
                avg_length += info["env_steps"]
                avg_collision_reward += info["collision_reward"]
                avg_collision_avoidance_reward += info["collision_avoidance_reward"]
                avg_pf_reward += info["path_following_reward"]
                avg_close_to_target_reward += info["close_to_target_reward"]


            avg_reward /= n_done
            avg_length /= n_done
            avg_collision_reward /= n_done
            avg_collision_avoidance_reward /= n_done
            avg_pf_reward /= n_done
            avg_close_to_target_reward /= n_done

            # Write to the tensorboard logger
            self.logger.record("episodes/avg_reward", avg_reward)
            self.logger.record("episodes/avg_length", avg_length)
            self.logger.record("episodes/avg_collision_reward", avg_collision_reward)
            self.logger.record("episodes/avg_collision_avoidance_reward", avg_collision_avoidance_reward)
            self.logger.record("episodes/avg_pf_reward", avg_pf_reward)
            self.logger.record("episodes/avg_close_to_target_reward", avg_close_to_target_reward)

        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass