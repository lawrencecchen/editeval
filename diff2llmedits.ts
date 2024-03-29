import { openai } from "./openai";

// https://github.com/tinygrad/tinygrad/pull/3994/files
const diff1 = `\
diff --git a/test/test_randomness.py b/test/test_randomness.py
index d086442082ff..66263b166602 100644
--- a/test/test_randomness.py
+++ b/test/test_randomness.py
@@ -1,12 +1,15 @@
-import math
-import unittest
+import unittest, math
 from functools import partial
 
 import numpy as np
 import torch
-from tinygrad import nn, dtypes, Tensor
+from tinygrad import nn, dtypes, Tensor, Device
 from tinygrad.helpers import THREEFRY
 from test.helpers import is_dtype_supported
+from hypothesis import given, settings, strategies as strat
+
+settings.register_profile("my_profile", max_examples=200, deadline=None)
+settings.load_profile("my_profile")
 
 # https://gist.github.com/devries/11405101
 def ksprob(a):
@@ -100,6 +103,22 @@ def test_randn(self):
     self.assertTrue(normal_test(Tensor.randn))
     self.assertTrue(equal_distribution(Tensor.randn, torch.randn, lambda x: np.random.randn(*x)))
 
+  @given(strat.sampled_from([dtypes.float, dtypes.float16]))#, dtypes.bfloat16]))  # TODO: add bfloat16
+  @unittest.skipIf(Device.DEFAULT=="RHIP", "broken in HIP CI")
+  def test_randn_finite(self, default_float):
+    if not is_dtype_supported(default_float): return
+    old_default_float = dtypes.default_float
+    # low precision can result in inf from randn
+    dtypes.default_float = default_float
+    t = Tensor.randn(1024, 1024)
+    mx = t.max().numpy().item()
+    mn = t.min().numpy().item()
+    if default_float == dtypes.float or (default_float == dtypes.float16 and not THREEFRY.value):
+      print(f"testing with {default_float=}")
+      assert math.isfinite(mx), mx
+      assert math.isfinite(mn), mn
+    dtypes.default_float = old_default_float
+
   def test_randint(self):
     self.assertFalse(normal_test(Tensor.randint))
     self.assertTrue(equal_distribution(partial(Tensor.randint, low=-2, high=5), numpy_func=lambda x: np.random.randint(low=-2, high=5, size=x)))`;

async function diff2edits(diff: string) {
	const response = await openai.chat.completions.create({
		model: "gpt-4-0125-preview",
		temperature: 0,
		messages: [
			{
				role: "user",
				content: `Diff:
${diff1}

Instruct a human to make the changes. Use language like:

After the function X, add the following code:
\`\`\`python
...
\`\`\`

or 

Delete the function Y, etc.

Don't write lists, just write paragraphs of instructions with  markdown code blocks interspersed.`,
				// Given the above diff, enumerate in natural language the changes made. For each change, describe the file affected, the line numbers, as well as the specific changes made. The goal is to create evals to test whether a language model can convert natural language descriptions of code changes into code diffs. Only output a list of changes with no preamble or extra information. Write in present tense, as if you were instructing someone to make the changes.`,
			},
		],
		// stream: true,
	});

	// for await (const message of response) {
	// 	const content = message.choices[0].delta.content;
	// 	if (content) {
	// 		process.stdout.write(content);
	// 	}
	// }
	const content = response.choices[0].message.content;
	if (!content) {
		throw new Error("No content in response");
	}
	return content;
}
