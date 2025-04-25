from langchain_community.document_loaders import PDFPlumberLoader

class LoadDocuments:

    def __init__(self):
        pass

    def load_pdf(self, path:str):
        document_loader = PDFPlumberLoader(path)
        file = document_loader.load()
        return [doc.page_content for doc in file]
    