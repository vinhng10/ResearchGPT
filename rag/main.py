import uuid
import qdrant_client
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from fastapi import FastAPI, UploadFile
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core import (
    VectorStoreIndex,
    Settings,
    SimpleDirectoryReader,
    ChatPromptTemplate,
)
from llama_index.core.schema import QueryBundle
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore


class Document(BaseModel):
    content: str


class Chat(BaseModel):
    content: str


class Message(BaseModel):
    author: str
    chat: str | None = None
    context: str | None = None
    content: str


class Environment(BaseSettings):
    llm_url: str = "https://development-llm.dbinno.com/v1"
    embed_model: str = "BAAI/bge-large-en-v1.5"
    vector_store_url: str = "https://development-qdrant.dbinno.com"
    temperature: float = 0.1
    similarity_top_k: int = 1
    chunk_size: int = 316
    chunk_overlap: int = 64


env = Environment()
app = FastAPI()

origins = [
    "https://development-app.dbinno.com",
    "https://development-api.dbinno.com",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Settings.llm = OpenAI(
    api_base=env.llm_url, temperature=env.temperature, max_tokens=1024
)
Settings.embed_model = FastEmbedEmbedding(
    model_name=env.embed_model, cache_dir="/storage"
)

# Sync Qdrant client:
client = qdrant_client.QdrantClient(url=env.vector_store_url, port=443)
vector_store = QdrantVectorStore(client=client, collection_name="Store")
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store, insert_batch_size=32
)
# Text Q&A prompt
text_qa_system_prompt = ChatMessage(
    content=(
        "<<SYS>> You are an expert Q&A system that is trusted around the world.\n"
        "You always answer the query using the provided context information "
        "and never use your prior knowledge. <</SYS>>"
    ),
    role=MessageRole.SYSTEM,
)

text_qa_prompt_tmpl_msgs = [
    text_qa_system_prompt,
    ChatMessage(
        content=(
            "Context information is below.\n"
            "---------------\n"
            "{context_str}\n"
            "---------------\n"
            "Given the context information and not prior knowledge, answer the query.\n"
            "Do not include statements like 'According to the provided text' or 'Based "
            "on the provided information' or anything similar in your answer."
            "Query: {query_str}\n"
            "Answer: "
        ),
        role=MessageRole.USER,
    ),
]

chat_text_qa_prompt = ChatPromptTemplate(message_templates=text_qa_prompt_tmpl_msgs)

query_engine = index.as_query_engine(
    streaming=True,
    similarity_top_k=env.similarity_top_k,
    text_qa_template=chat_text_qa_prompt,
)
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=env.chunk_size, chunk_overlap=env.chunk_overlap)
    ],
    vector_store=vector_store,
)

# Async Qdrant client:
async_client = qdrant_client.AsyncQdrantClient(url=env.vector_store_url, port=443)
async_client.set_model(env.embed_model, cache_dir="/storage")


@app.get("/", response_class=HTMLResponse)
async def ui():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>Simple Chat Application with File Uploader</title>
        <link
        rel="stylesheet"
        href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css"
        />
        <style>
        body {
            margin: 0;
            padding: 0;
        }
        .container {
            display: flex;
            flex-direction: row;
            height: 100vh;
        }
        .column {
            flex: 1;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            padding: 0px 10px 0px 10px;
        }
        .messages {
            overflow-y: auto;
            flex-grow: 1;
        }
        .message-container {
            margin-bottom: 10px;
            white-space: pre-wrap;
            color: #fff;
            border-radius: 10px;
            padding: 10px;
        }
        .user-message {
            background-color: #007bff;
        }
        .ai-message {
            background-color: #28a745;
        }
        .document-message {
            background-color: #949400;
        }
        .input-container {
            margin-top: 10px;
            margin-bottom: 10px;
        }
        .file-upload-column {
            border-right: 1px solid #ccc;
        }
        </style>
    </head>
    <body>
        <div class="container">
        <div class="column file-upload-column">
            <div>
            <h3>Document Upload</h3>
            <p>
                The uploaded documents will be stored in database and "learned" by
                AI.
            </p>
            <input type="file" id="fileInput" multiple />
            <button id="submitBtn" class="btn btn-primary mt-2">Submit</button>
            <ul id="fileList"></ul>
            </div>
            <hr />
            <h3>Retrieved Information</h3>
            <p>
            This is just for understanding what information the AI is able to
            retrieve from the database. In practice, these are not displayed for
            end user.
            </p>
            <div id="retrievedDocuments" class="messages"></div>
        </div>
        <div class="column file-upload-column">
            <h3>Ask Question About Uploaded Documents</h3>
            <div class="messages" id="ragMessages">
            <!-- Messages will be added dynamically here -->
            </div>
            <div class="input-container">
            <input
                type="text"
                class="form-control"
                id="ragInput"
                placeholder="Ask AI anything about your documents ..."
            />
            </div>
        </div>
        <div class="column">
            <h3>General Purpose AI Assistant</h3>
            <p>
            Suitable for general tasks that don't need information from uploaded
            documents.
            </p>
            <div class="messages" id="aiMessages">
            <!-- Messages will be added dynamically here -->
            </div>
            <div class="input-container">
            <input
                type="text"
                class="form-control"
                id="aiInput"
                placeholder="Ask AI anything ..."
            />
            </div>
        </div>
        </div>

        <script>
        const ragMessagesContainer = document.getElementById("ragMessages");
        const aiMessagesContainer = document.getElementById("aiMessages");
        const ragInput = document.getElementById("ragInput");
        const aiInput = document.getElementById("aiInput");
        const aiMessages = [
            {
            role: "system",
            content: "<<SYS>> You are a helpful assistant. <</SYS>>",
            },
        ];

        async function retrieve(message) {
            const url = "/retrieve";

            const data = {
            content: message,
            };

            try {
            const response = await fetch(url, {
                method: "POST",
                headers: {
                "Content-Type": "application/json",
                },
                body: JSON.stringify(data),
            });
            const documents = await response.json();
            const retrievedDocuments =
                document.getElementById("retrievedDocuments");
            retrievedDocuments.innerHTML = "";

            documents.forEach((doc) => {
                const element = document.createElement("p");
                element.classList.add("message-container", "document-message");
                element.textContent = doc.content;
                retrievedDocuments.appendChild(element);
            });
            } catch (error) {
            console.error(error);
            }
        }

        async function ragChat(message) {
            const url = "/chat";

            const data = {
            content: message,
            };

            const response = await fetch(url, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(data),
            });

            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            const messageElement = document.createElement("div");
            messageElement.classList.add("message-container", "ai-message");
            ragMessagesContainer.appendChild(messageElement);

            let chunks = "";

            while (true) {
            const { done, value } = await reader.read();

            if (done) break;
            try {
                chunks += decoder.decode(value, {
                stream: true,
                });
                messageElement.textContent = chunks;
                scrollToBottom(ragMessagesContainer);
            } catch (error) {
                continue;
            }
            }
        }

        async function aiChat(input) {
            const url = "https://development-llm.dbinno.com/v1/chat/completions";

            aiMessages.push({
            role: "user",
            content: input,
            });

            const data = {
            messages: aiMessages,
            stream: true,
            max_tokens: 1024,
            temperature: 0.8,
            };

            const response = await fetch(url, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(data),
            });
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            const messageElement = document.createElement("div");
            messageElement.classList.add("message-container", "ai-message");
            aiMessagesContainer.appendChild(messageElement);
            let answer = "";

            while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            try {
                const chunks = decoder
                .decode(value, {
                    stream: true,
                })
                .trim()
                .replace(/[\\r\\n]+/g, "\\r\\n")
                .split("\\r\\n");

                for (const chunk of chunks) {
                const content = JSON.parse(chunk.substring(6)).choices[0].delta;
                if (content.hasOwnProperty("content")) {
                    answer += content.content;
                    messageElement.textContent = answer;
                    scrollToBottom(aiMessagesContainer);
                }
                }
            } catch (error) {
                console.error("===>", error);
                continue;
            }
            }

            aiMessages.push({
            role: "assistant",
            content: answer.slice(0, -4),
            });
        }

        function scrollToBottom(container) {
            container.scrollTop = container.scrollHeight;
        }

        ragInput.addEventListener("keypress", function (e) {
            if (e.key === "Enter") {
            const message = ragInput.value.trim();
            if (message !== "") {
                const messageElement = document.createElement("div");
                messageElement.classList.add("message-container", "user-message");
                messageElement.textContent = message;
                ragMessagesContainer.appendChild(messageElement);
                scrollToBottom(ragMessagesContainer);
                retrieve(message);
                ragChat(message);
                ragInput.value = "";
            }
            }
        });

        aiInput.addEventListener("keypress", function (e) {
            if (e.key === "Enter") {
            const message = aiInput.value.trim();
            if (message !== "") {
                const messageElement = document.createElement("div");
                messageElement.classList.add("message-container", "user-message");
                messageElement.textContent = message;
                aiMessagesContainer.appendChild(messageElement);
                scrollToBottom(aiMessagesContainer);
                aiChat(message);
                aiInput.value = "";
            }
            }
        });

        // File upload handling
        const fileInput = document.getElementById("fileInput");
        const fileList = document.getElementById("fileList");
        const submitBtn = document.getElementById("submitBtn");

        submitBtn.addEventListener("click", async function () {
            submitBtn.disabled = true;
            submitBtn.textContent = "Uploading...";

            const files = Array.from(fileInput.files);

            const formData = new FormData();
            files.forEach((file) => {
            formData.append("files", file);
            const listItem = document.createElement("li");
            listItem.textContent = file.name;
            fileList.appendChild(listItem);
            });

            try {
            const response = await fetch("/upload", {
                method: "POST",
                body: formData,
            });
            } catch (error) {
            console.error("Error uploading files:", error);
            } finally {
            // Enable the button and change its text back to "Submit"
            submitBtn.disabled = false;
            submitBtn.textContent = "Submit";
            }
        });
        </script>
    </body>
    </html>
    """


@app.post("/upload")
async def upload_files(files: list[UploadFile]):
    filepaths = []
    for file in files:
        filepath = f"/storage/{file.filename}"
        filepaths.append(filepath)
        with open(filepath, "wb") as f:
            f.write(await file.read())
    documents = SimpleDirectoryReader(input_files=filepaths).load_data()
    nodes = pipeline.run(documents=documents)
    index.insert_nodes(nodes)
    return {"filenames": [file.filename for file in files]}


@app.post("/retrieve")
async def retrieve(chat: Chat) -> list[Document]:
    nodes = query_engine.retrieve(QueryBundle(query_str=chat.content))
    data = [Document(content=node.text) for node in nodes]
    return data


@app.post("/embeddings")
async def embeddings(message: Message):
    if message.chat:
        content = (
            f"Conversation with {message.chat}\n{message.author}: {message.content}"
        )
    elif message.context:
        content = (
            f"Conversation about {message.context}\n{message.author}: {message.content}"
        )
    response = await async_client.add(
        collection_name="One", documents=[content], ids=[str(uuid.uuid4())]
    )
    return response


@app.post("/chat")
async def chat(chat: Chat):

    def stream():
        response = query_engine.query(chat.content)
        for text in response.response_gen:
            yield text

    return StreamingResponse(stream(), media_type="text/event-stream")
