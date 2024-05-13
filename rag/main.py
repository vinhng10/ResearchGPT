import qdrant_client
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from fastapi import FastAPI, UploadFile
from fastapi.responses import StreamingResponse, HTMLResponse
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


class Settings(BaseSettings):
    embed_model: str = "BAAI/bge-large-en-v1.5"
    vector_store_url: str = "https://development-qdrant.dbinno.com"
    temperature: float = 0.8
    similarity_top_k: int = 1
    chunk_size: int = 512
    chunk_overlap: int = 64


settings = Settings()
app = FastAPI()

Settings.llm = OpenAI(
    api_base="https://development-llm.dbinno.com/v1", temperature=0.1, max_tokens=2048
)
Settings.embed_model = FastEmbedEmbedding(model_name=settings.embed_model)
client = qdrant_client.QdrantClient(url=settings.vector_store_url, port=443)
vector_store = QdrantVectorStore(client=client, collection_name="Store")
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store, insert_batch_size=32
)
# text qa prompt
text_qa_system_prompt = ChatMessage(
    content=(
        "<SYS> You are an expert Q&A system that is trusted around the world.\n"
        "Always answer the query using the provided context information, "
        "and not prior knowledge.\n"
        "Some rules to follow:\n"
        "1. Never directly reference the given context in your answer.\n"
        "2. Avoid statements like 'Based on the context, ...' or "
        "'The context information ...' or anything along "
        "those lines. </SYS>"
    ),
    role=MessageRole.SYSTEM,
)

text_qa_prompt_tmpl_msgs = [
    text_qa_system_prompt,
    ChatMessage(
        content=(
            "Context information is below.\n"
            "-----\n"
            "{context_str}\n"
            "-----\n"
            "Given the context information and not prior knowledge, "
            "answer the query.\n"
            "Query: {query_str}\n"
            "Answer: "
        ),
        role=MessageRole.USER,
    ),
]

chat_text_qa_prompt = ChatPromptTemplate(message_templates=text_qa_prompt_tmpl_msgs)

query_engine = index.as_query_engine(
    streaming=True,
    similarity_top_k=settings.similarity_top_k,
    text_qa_template=chat_text_qa_prompt,
)
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(
            chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
        )
    ],
    vector_store=vector_store,
)


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
            <h2>File Upload</h2>
            <input type="file" id="fileInput" multiple />
            <button id="submitBtn" class="btn btn-primary mt-2">Submit</button>
            <ul id="fileList"></ul>
            </div>

            <h2>Retrieved Information</h2>
            <h6>(Hidden From Users)</h6>
            <div id="retrievedDocuments" class="messages"></div>
        </div>
        <div class="column">
            <h2>Conversation</h2>
            <div class="messages" id="messages">
            <!-- Messages will be added dynamically here -->
            </div>
            <div class="input-container">
            <input
                type="text"
                class="form-control"
                id="messageInput"
                placeholder="Type your message..."
            />
            </div>
        </div>
        </div>

        <script>
        const messagesContainer = document.getElementById("messages");
        const messageInput = document.getElementById("messageInput");

        async function retrieve(message) {
            const url = "http://localhost:8000/retrieve/";

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

        async function aiChat(message) {
            const url = "http://localhost:8000/chat/";

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
            messagesContainer.appendChild(messageElement);

            let chunks = "";

            while (true) {
            const { done, value } = await reader.read();

            if (done) break;
            try {
                chunks += decoder.decode(value, {
                stream: true,
                });
                messageElement.textContent = chunks;
                scrollToBottom();
            } catch (error) {
                continue;
            }
            }
        }

        function scrollToBottom() {
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }

        messageInput.addEventListener("keypress", function (e) {
            if (e.key === "Enter") {
            const message = messageInput.value.trim();
            if (message !== "") {
                const messageElement = document.createElement("div");
                messageElement.classList.add("message-container", "user-message");
                messageElement.textContent = message;
                messagesContainer.appendChild(messageElement);
                scrollToBottom();
                retrieve(message);
                aiChat(message);
                messageInput.value = "";
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


@app.post("/upload/")
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


@app.post("/retrieve/")
async def retrieve(chat: Chat) -> list[Document]:
    nodes = query_engine.retrieve(QueryBundle(query_str=chat.content))
    data = [Document(content=node.text) for node in nodes]
    return data


@app.post("/chat/")
async def chat(chat: Chat):

    def stream():
        response = query_engine.query(chat.content)
        for text in response.response_gen:
            yield text

    return StreamingResponse(stream(), media_type="text/event-stream")