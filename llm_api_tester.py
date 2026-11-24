import sys

import truststore
truststore.inject_into_ssl()

from dotenv import load_dotenv

from safechain.core.model import model

sys.stdout.reconfigure(encoding="utf-8")


load_dotenv(".env", override=True)


GPT_MODEL = model("3")

response = GPT_MODEL.invoke("hello")

print(response.content)


EMBEDDING_MODEL = model("4")

output = EMBEDDING_MODEL.embed_documents(["hello"])[0]
print(len(output))
