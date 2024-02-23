import os
import numpy as np
from bigcode_eval.base import Task
from bigcode_eval.tasks.custom_metrics.execute import check_correctness
from concurrent.futures import ProcessPoolExecutor

def make_cot_output_prompt(s):
    code, input = s
    return f"""You are given a Python function and an assertion containing an input to the function. Complete the assertion with a literal (no unsimplified expressions, no function calls) containing the output when executing the provided code on the given input, even if the function is incorrect or incomplete. Do NOT output any extra information. Execute the program step by step before arriving at an answer, and provide the full assertion with the correct output in [ANSWER] and [/ANSWER] tags, following the examples.

[PYTHON]
def f(s):
    s = s + s
    return "b" + s + "a"
assert f("hi") == ??
[/PYTHON]
[THOUGHT]
Let's execute the code step by step:

1. The function f is defined, which takes a single argument s.
2. The function is called with the argument "hi", so within the function, s is initially "hi".
3. Inside the function, s is concatenated with itself, so s becomes "hihi".
4. The function then returns a new string that starts with "b", followed by the value of s (which is now "hihi"), and ends with "a".
5. The return value of the function is therefore "bhihia".
[/THOUGHT]
[ANSWER]
assert f("hi") == "bhihia"
[/ANSWER]

[PYTHON]
{code}
assert f({input}) == ??
[/PYTHON]
[THOUGHT]
"""

def make_direct_output_prompt(s):
    code, input = s
    return f"""You are given a Python function and an assertion containing an input to the function. Complete the assertion with a literal (no unsimplified expressions, no function calls) containing the output when executing the provided code on the given input, even if the function is incorrect or incomplete. Do NOT output any extra information. Provide the full assertion with the correct output in [ANSWER] and [/ANSWER] tags, following the examples.

[PYTHON]
def f(n):
    return n
assert f(17) == ??
[/PYTHON]
[ANSWER]
assert f(17) == 17
[/ANSWER]

[PYTHON]
def f(s):
    return s + "a"
assert f("x9j") == ??
[/PYTHON]
[ANSWER]
assert f("x9j") == "x9ja"
[/ANSWER]

[PYTHON]
{code}
assert f({input}) == ??
[/PYTHON]
[ANSWER]
"""

def make_direct_input_prompt(s):
    code, output = s
    return f"""You will be given a function f and an output in the form f(??) == output. Find any input such that executing f on the input leads to the given output. There may be multiple answers, but you should only output one. In [ANSWER] and [/ANSWER] tags, complete the assertion with one such input that will produce the output when executing the function.

[PYTHON]
def f(my_list):
    count = 0
    for i in my_list:
        if len(i) % 2 == 0:
            count += 1
    return count
assert f(??) == 3
[/PYTHON]
[ANSWER]
assert f(["mq", "px", "zy"]) == 3
[/ANSWER]

[PYTHON]
def f(s1, s2):
    return s1 + s2
assert f(??) == "banana"
[/PYTHON]
[ANSWER]
assert f("ba", "nana") == "banana"
[/ANSWER]

[PYTHON]
{code}
assert f(??) == {output}
[/PYTHON]
[ANSWER]
"""

def make_cot_input_prompt(s):
    code, output = s
    return f"""You will be given a function f and an output in the form f(??) == output. Your task is to find any input such that executing f on the input leads to the given output. There may be multiple answers, but only output one. First, think step by step. You MUST surround the answer with [ANSWER] and [/ANSWER] tags. Express your answer as a passing assertion containing the input and the given output.

[PYTHON]
def f(x):
    return x + 1
assert f(??) == 17
[/PYTHON]
[THOUGHT]
To find an input such that executing f on the input leads to the given output, we can work backwards from the given assertion. We know that f(??) == 17. 

Since the function f(x) returns x + 1, for f(??) to be equal to 17, the value of ?? should be 16. 
[/THOUGHT]
[ANSWER]
assert f(16) == 17
[/ANSWER]

[PYTHON]
{code}
assert f(??) == {output}
[/PYTHON]
[THOUGHT]
"""

def pass_at_k(n, c, k):
    if n - c < k: return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

def evaluate_score(args):
    gs, (c, i, o), mode = args

    execution_results = []
    for g in gs:
        if mode == "input" and "f(" not in g:
            execution_results.append({"passed": False})
        elif mode == "output" and f"f({i})" in g:
            execution_results.append({"passed": False})
        else:
            code_to_execute = f"{c}\nassert {o} == {g}"
            execution_results.append(check_correctness(code_to_execute, 3, None, None))
    return [i["passed"] for i in execution_results]

def create_all_tasks():
    def create_task(mode, cot):
        class CRUXEval(GeneralCRUXEval):
            def __init__(self, **kwargs):
                super().__init__(mode, cot, **kwargs)
        return CRUXEval
    return {
        "cruxeval-input-cot": create_task("input", True),
        "cruxeval-input": create_task("input", False),
        "cruxeval-output-cot": create_task("output", True),
        "cruxeval-output": create_task("output", False),
    }

class GeneralCRUXEval(Task):
    DATASET_PATH = "cruxeval-org/cruxeval"
    DATASET_NAME = None

    def __init__(self, mode, cot):
        self.mode = mode
        self.cot = cot
        super().__init__(
            stop_words=["[/ANSWER]"],
            requires_execution=True,
        )

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset["test"]

    def get_prompt(self, doc):
        if self.mode == "input":
            if self.cot:
                return make_cot_input_prompt((doc["code"], doc["output"]))
            else:
                return make_direct_input_prompt((doc["code"], doc["output"]))
        elif self.mode == "output":
            if self.cot:
                return make_cot_output_prompt((doc["code"], doc["input"]))
            else:
                return make_direct_output_prompt((doc["code"], doc["input"]))
        else:
            raise ValueError(f"Unknown mode {self.mode}")

    def get_reference(self, doc):
        return (doc["code"], doc["input"], doc["output"])

    def postprocess_generation(self, generation, idx):
        prompt = self.get_prompt(self.get_dataset()[idx])
        assert generation.startswith(prompt)
        generation = generation[len(prompt):]

        if self.cot:
            if "[ANSWER]" in generation:
                generation = generation.split("[ANSWER]")[1].strip()

        # format: assert f(xxx) == yyy
        if "[/ANSWER]" in generation:
            generation = generation.split("[/ANSWER]")[0].strip()
        if self.mode == "input": 
            if "==" in generation:
                generation = generation.split("==")[0].strip()
            if "assert" in generation:
                generation = generation.split("assert")[1].strip()
        elif self.mode == "output":
            if "==" in generation:
                generation = generation.split("==")[1].strip()

        return generation.strip()


    def process_results(self, generations, references):
        args_list = zip(generations, references, [self.mode] * len(generations))
        max_workers = max(1, os.cpu_count() - 10)
        with ProcessPoolExecutor(max_workers) as executor:
            results = executor.map(evaluate_score, args_list)
        all_scores = list(results)

        # Compute pass@k scores
        pass_at_1s, pass_at_5s = [], []
        for execution_result in all_scores:
            c, n = execution_result.count(True), len(execution_result)
            pass_at_1s.append(pass_at_k(n, c, 1))
            pass_at_5s.append(pass_at_k(n, c, 5))

        return {"raw_generations": generations,
                "raw_scored_generations": {f"sample_{i}": all_scores[i] for i in range(len(generations))},
                "pass_at_1": sum(pass_at_1s) / len(pass_at_1s) * 100,
                "pass_at_5": sum(pass_at_5s) / len(pass_at_5s) * 100}