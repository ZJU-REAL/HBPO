"""
This module contains the RewardMathFn class, which evaluates mathematical answers
and assigns rewards based on their correctness. It utilizes a language model to 
validate answers when necessary.
"""
from typing import List, Union

from rllm.globals import THOUGHT_DELIMITER_START, THOUGHT_DELIMITER_END, OAI_RM_MODEL
from rllm.rewards import RewardConfig, RewardFn, RewardInput, RewardOutput, RewardType
from rllm.rewards.math_utils.utils import extract_answer, grade_answer_sympy, grade_answer_mathd

from rllm.system_prompts import ORM_PROMPT
from rllm.utils import call_gemini_llm, call_oai_rm_llm
import json 

import numpy as np
import re

ORM_USER_TEMPLATE = """
Problem: {problem}
Answer 1: {answer_1}
Answer 2: {answer_2}
"""

class RewardMathFn(RewardFn):
    """
    Reward function for evaluating mathematical answers.

    This class implements the __call__ method to process the input and determine
    the reward based on the correctness of the provided answer compared to the ground truth.
    """

    def __call__(self, input: RewardInput) -> RewardOutput:
        assert input.problem_type == RewardType.MATH, \
            "Invalid problem type: expected 'MATH', but got '{}'".format(input.problem_type)
        
        problem = input.problem
        model_response = input.model_response
        valid_response_length = input.valid_response_length
        pattern = r"<think> I will answer the question with (\d+) tokens"
        match = re.search(pattern, model_response)
        # Extract solution.
        if THOUGHT_DELIMITER_END in model_response: #强制有</think>标签
            model_solution = model_response.split(THOUGHT_DELIMITER_END)[1]
        else:
            # if match:
            #     number = int(match.group(1))
            #     return RewardOutput(reward=self.config.format_error_reward-1 / 4096 * abs(number - valid_response_length), is_correct=False).to_dict()
            # else:
            return RewardOutput(reward=self.config.format_error_reward, is_correct=False).to_dict()
        
        model_answer = extract_answer(model_solution)
        if model_answer is None:
            # if match:
            #     number = int(match.group(1))
            #     return RewardOutput(reward=self.config.format_error_reward-1 / 4096 * abs(number - valid_response_length), is_correct=False).to_dict()
           
            return RewardOutput(reward=self.config.format_error_reward, is_correct=False).to_dict()

        # Process the ground truth(s)
        ground_truths = input.metadata.get("answer", None)
        if ground_truths is None:
            # if match:
            #     number = int(match.group(1))
            #     return RewardOutput(reward=self.config.unk_error_reward-1 / 4096 * abs(number - valid_response_length), is_correct=False).to_dict()

            
            return RewardOutput(reward=self.config.unk_error_reward, is_correct=False).to_dict()
        
        # Convert single answer to list for uniform processing
        if isinstance(ground_truths, (str, float, int)):
            ground_truths = [ground_truths]
            
        # Process each ground truth
        processed_ground_truths = []
        for truth in ground_truths:
            truth = str(truth)
            if "\\boxed" in truth:
                processed_truth = extract_answer(truth)
                if processed_truth is not None:
                    processed_ground_truths.append(processed_truth)
            else:
                processed_ground_truths.append(truth)
        
        if not processed_ground_truths:
            # if match:
            #     number = int(match.group(1))
            #     return RewardOutput(reward=self.config.unk_error_reward-1 / 4096 * abs(number - valid_response_length), is_correct=False).to_dict()
            return RewardOutput(reward=self.config.unk_error_reward, is_correct=False).to_dict()

        # Check against all possible correct answers
        for ground_truth in processed_ground_truths:
            is_correct = grade_answer_mathd(model_answer, ground_truth) or grade_answer_sympy(model_answer, ground_truth)
            if is_correct and valid_response_length <= 4096:
                # Apply tool call bonus if applicable and answer is correct
                '''hierarchical reward function'''
                PI_APPROX = 3.1416
                result = 4 * np.cos(PI_APPROX * valid_response_length / 8192)
                retval = np.round(result, 2)
                if match:
                    number = int(match.group(1))
                    standard = 4 * np.cos(PI_APPROX * number / 8192)
                    retval = min(result - 1 / 4096 * abs(number - valid_response_length), standard)
                    # if number == 3072:
                    #     retval = np.round(result, 2)
                    #retval = 1  #classic reward
                reward = retval #self.config.correct_reward
                #reward=1
                if input.metadata.get("has_toolcall", False):
                    reward += self.config.toolcall_bonus
                return RewardOutput(reward=reward, is_correct=True).to_dict()

        # If latex heuristics fail and ORM is enabled, use LLM as ORM to evaluate correctness
        if self.config.use_math_orm:
            
            for ground_truth in processed_ground_truths:
                
                try:
                    orm_response = call_gemini_llm(
                        system_prompt=ORM_PROMPT,
                        prompt=ORM_USER_TEMPLATE.format(problem=problem, answer_1=model_answer, answer_2=ground_truth),
                        temperature=0.0,
                    )

                    if "[[YES]]" in orm_response:
                        return RewardOutput(reward=self.config.correct_reward, is_correct=True).to_dict()
                except Exception as e:
                    print ("Error calling Gemini ORM, trying OAI RM")
                    orm_response = call_oai_rm_llm(
                        system_prompt=ORM_PROMPT,
                        prompt=ORM_USER_TEMPLATE.format(problem=problem, answer_1=model_answer, answer_2=ground_truth),
                        temperature=0.0,
                        model_id=OAI_RM_MODEL,
                    )
                    
                    if "[[YES]]" in orm_response:
                        return RewardOutput(reward=self.config.correct_reward, is_correct=True).to_dict()
                    continue
        # if match:
        #     number = int(match.group(1))
        #     return RewardOutput(reward=self.config.incorrect_reward-1 / 4096 * abs(number - valid_response_length), is_correct=False).to_dict()        
        return RewardOutput(reward=self.config.incorrect_reward, is_correct=False).to_dict()



def rllm_reward_fn_math(data_source: str, llm_solution: str, ground_truth: Union[str, List[str]],valid_response_length:int, extra_info={}, **kwargs):
    """Evaluates mathematical solutions against ground truth answers.

    This function creates a reward function to evaluate mathematical solutions by comparing
    them against provided ground truth answers. It can optionally use a language model
    for more sophisticated answer validation.

    Args:
        data_source: The source/dataset the problem comes from
        llm_solution: The solution string provided by the language model to evaluate
        ground_truth: Either a single string or list of strings containing valid answers
        enable_llm: Whether to enable language model validation for complex cases (default: False)

    Returns:
        bool: True if the solution is deemed correct, False otherwise

    Example:
        >>> rllm_reward_fn_math("gsm8k", "x = 5", "5", False)
        True
    """
    reward_config = RewardConfig()
    reward_fn = RewardMathFn(reward_config)
    reward_response = reward_fn(RewardInput(problem=None,
                                            problem_type=RewardType.MATH,
                                            model_response=llm_solution,
                                            valid_response_length=valid_response_length,
                                            metadata={"answer": ground_truth, **extra_info},
                                            data_source=data_source))
    return reward_response


if __name__ == "__main__":
    reward = RewardMathFn(RewardConfig)
    test_input = RewardInput(
        data_source="",
        problem=(
            "Let $P(x)=x^{4}+2 x^{3}-13 x^{2}-14 x+24$ be a polynomial with roots "
            "$r_{1}, r_{2}, r_{3}, r_{4}$. Let $Q$ be the quartic polynomial with roots "
            "$r_{1}^{2}, r_{2}^{2}, r_{3}^{2}, r_{4}^{2}$, such that the coefficient "
            "of the $x^{4}$ term of $Q$ is 1. Simplify the quotient $Q\\left(x^{2}\\right) / P(x)$, "
            "leaving your answer in terms of $x$. (You may assume that $x$ is not equal to "
            "any of $\\left.r_{1}, r_{2}, r_{3}, r_{4}\\right)$."
        ),
        problem_type=RewardType.MATH,
        model_response=(
            "<think>...</think>\nThe answer is \\boxed{24 + 14*x + (-13)*x^2 - 2*x^3 + x^4}."
        ),
        metadata={"answer": ["10", "$x^{4}-2 x^{3}-13 x^{2}+14 x+24$"], "has_toolcall": True}
    )
    output = reward(test_input)
    print(output)
