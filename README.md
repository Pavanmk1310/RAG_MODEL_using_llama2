The program employs the Llama 2 model as the core intelligence driving the Retrieval-Augmented Generation (RAG) system, utilizing Ollama for model invocation. For embedding the provided PDFs, Ollama's embedding capabilities are leveraged, with the embeddings being stored in RAM as a vector store. This process is facilitated through DocArray, enabling efficient handling and storage of the document embeddings. FAISS is used to retrieve relevant embeddings, ensuring optimal performance in similarity search and document retrieval tasks.

The program is highly dependent on hardware resources, which may result in longer execution times. However, if the program is connected to a GPU, it can potentially achieve faster execution by utilizing the CUDA cores of an NVIDIA GPU, significantly accelerating processing tasks such as model inference and embedding operations.
