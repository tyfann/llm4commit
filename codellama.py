
from llama_cpp import Llama
from ctransformers import AutoModelForCausalLM, AutoTokenizer


model = "../../../../llama.cpp/models/codellama/codellama-7b.Q4_K_M.gguf"
## Imports


# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
llm = AutoModelForCausalLM.from_pretrained(model_path_or_repo_id=model, model_type="llama", gpu_layers=50)

prompts = ["The following is a diff which describes the code changes in a commit, Your task is to write a short commit message accordingly. diff --git a/src/gwt/src/org/rstudio/studio/client/workbench/views/source/editors/text/TextEditingTargetRenameHelper.java b/src/gwt/src/org/rstudio/studio/client/workbench/views/source/editors/text/TextEditingTargetRenameHelper.java\nindex e8d986667..ce243ca1f 100644\n--- a/src/gwt/src/org/rstudio/studio/client/workbench/views/source/editors/text/TextEditingTargetRenameHelper.java\n+++ b/src/gwt/src/org/rstudio/studio/client/workbench/views/source/editors/text/TextEditingTargetRenameHelper.java\n@@ -44,7 +44,7 @@ public class TextEditingTargetRenameHelper\n       \n       editor_.setCursorPosition(position);\n       \n-      // Validate that we're looking at an R identifier (TODO: refactor strings?)\n+      // Validate that we're looking at an R identifier\n       String targetValue = cursor.currentValue();\n       String targetType = cursor.currentType();\n       \n@@ -104,7 +104,6 @@ public class TextEditingTargetRenameHelper\n       }\n       \n       // Otherwise, just rename the variable within the current scope.\n-      // TODO: Do we need to look into parent scopes?\n       return renameVariablesInScope(\n             editor_.getCurrentScope(),\n             targetValue,\nAccording to the diff, the commit message should be:",
          "The following is a diff which describes the code changes in a commit, Your task is to write a short commit message accordingly. diff --git a/host-controller/src/main/java/org/jboss/as/host/controller/HostController.java b/host-controller/src/main/java/org/jboss/as/host/controller/HostController.java\nindex 04058ebdcc..42e15b8153 100644\n--- a/host-controller/src/main/java/org/jboss/as/host/controller/HostController.java\n+++ b/host-controller/src/main/java/org/jboss/as/host/controller/HostController.java\n@@ -230,7 +230,7 @@ public class HostController {\n             }\n         });\n \n-        final ServiceActivatorContext serviceActivatorContext = new ServiceActivatorContextImpl(batchBuilder);\n+        final ServiceActivatorContext serviceActivatorContext = new ServiceActivatorContextImpl(batchBuilder, serviceContainer);\n \n         // Always activate the management port\n         activateManagementCommunication(serviceActivatorContext);\n According to the diff, the commit message should be:"]

for prompt in prompts:
    print(llm(prompt, batch_size=8))