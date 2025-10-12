import torch
from configurations.configurations import LearningStrategyConfiguration
from learning_strategy.base_learning_strategy import BaseLearningStrategy

class DAPStrategy(BaseLearningStrategy):

    def __init__(self, critic_model, optimizer, configuration: LearningStrategyConfiguration, logger=None):
        """
        TODO: Provide description of the current strategy
        """
        super().__init__(critic_model, optimizer, configuration, logger)

        #TODO: Create a StrategySpecificEnums
        self._sigma = self._configuration.parameters.get("sigma", 120)
    
    def _calculate_loss(self, scaffold_batch, decorator_batch, score, actor_nlls):
        critic_nlls = self.critic_model.likelihood(*scaffold_batch, *decorator_batch)
        #print(f"Calculated Critic NLLs for batch of size {scaffold_batch[0].size(0)}")
        #print(f"Critic NLLs: {critic_nlls}")
        #assert False, "check critic nll"
        negative_critic_nlls = -critic_nlls
        negative_actor_nlls = -actor_nlls
        augmented_nlls = negative_critic_nlls + self._sigma * self._to_tensor(score)
        #print(f"score: {score}")
        #print(f"self.sigma: {self._sigma}")
        #print(f"Augmented NLLs: {augmented_nlls}")
        loss = torch.pow((augmented_nlls - negative_actor_nlls), 2)
        #print(f"Initial Losses: {loss}")
        loss = loss.mean()
        #print(f"Loss: {loss}")
        #assert False, "check loss"
        return loss, negative_actor_nlls, negative_critic_nlls, augmented_nlls