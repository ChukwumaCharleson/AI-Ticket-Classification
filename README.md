# AI-Ticket-Classification

This Chatbot application is designed to streamline information retrieval from PDF documents and provide efficient departmental routing for user inquiries. Leveraging Python, Huggingface models, Langchain, Chroma DB, and SVM ML model, this solution a basic framework for document analysis and department prediction.

## Features

1. **Load Data Store**: Easily upload PDF documents containing relevant information for analysis.

2. **Create ML Model**: Train, validate, and save a machine learning model using provided training data. This model predicts the appropriate department for each user question.

3. **Home Interface**: Engage with the chat interface to ask questions regarding the uploaded PDF content. If the chatbot cannot provide an answer, users can submit a ticket for further assistance.

4. **Pending Tickets Management**: Monitor and manage submitted tickets in the Pending Tickets section to track their resolution status.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your/repository.git
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Load Data Store**:
   - Navigate to the "Load Data Store" section.
   - Upload PDF documents containing relevant information.

2. **Create ML Model**:
   - Access the "Create ML Model" section.
   - Upload training data to train and validate the machine learning model.
   - Save the trained model for future use.

3. **Home Interface**:
   - Interact with the chat interface in the "Home" section.
   - Ask questions regarding the content of the uploaded PDF documents.
   - Click "Submit Ticket" if the chatbot cannot provide an answer.

4. **Pending Tickets Management**:
   - View and manage submitted tickets in the "Pending Tickets" section.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- Huggingface for their powerful models
- Langchain for language processing capabilities
- Chroma DB for efficient data storage and retrieval
