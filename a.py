import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
import ollama # استيراد مكتبة ollama
import os
import psutil

# تأكد من تحميل نموذج spacy المطلوب
# python -m spacy download en_core_web_sm
try:
    embeddings = SpacyEmbeddings(model_name="en_core_web_sm")
except OSError:
    st.error("لم يتم العثور على نموذج Spacy 'en_core_web_sm'. الرجاء تثبيته باستخدام: python -m spacy download en_core_web_sm")
    st.stop() # إيقاف التطبيق إذا لم يتم العثور على النموذج

# --- دوال معالجة PDF وتقسيمه (لا تغيير) ---
def pdf_read(pdf_doc):
    text = ""
    for pdf in pdf_doc:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        except Exception as e:
            st.error(f"خطأ في قراءة الملف {pdf.name}: {e}")
    return text

def create_text_chunks(text, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len # تحديد دالة الطول لتجنب التحذيرات المحتملة
    )
    text_chunks = text_splitter.split_text(text)
    return text_chunks

# --- دوال تخزين المتجهات والاسترجاع (تعديل وتحسين) ---
def create_and_save_vector_store(text_chunks, index_name="faiss_db"):
    """Create a vector store and save it locally."""
    try:
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local(index_name)
        st.success(f"تم إنشاء وحفظ تخزين المتجهات في '{index_name}'")
        return vector_store
    except Exception as e:
        st.error(f"فشل في إنشاء/حفظ تخزين المتجهات: {e}")
        return None

def load_vector_store(index_name="faiss_db"):
    """Load vector store from local."""
    if os.path.exists(index_name):
        try:
            # allow_dangerous_deserialization=True ضروري لتحميل FAISS المحفوظ محليًا
            vector_store = FAISS.load_local(index_name, embeddings, allow_dangerous_deserialization=True)
            st.info(f"تم تحميل تخزين المتجهات من '{index_name}'")
            return vector_store
        except Exception as e:
            st.error(f"فشل في تحميل تخزين المتجهات: {e}")
            return None
    else:
        st.warning(f"ملف تخزين المتجهات '{index_name}' غير موجود.")
        return None

def get_relevant_context(vector_store, question, k=4):
    """Retrieve relevant chunks based on the question."""
    if vector_store:
        try:
            retriever = vector_store.as_retriever(search_kwargs={"k": k})
            relevant_docs = retriever.invoke(question) # استخدام invoke بدلاً من get_relevant_documents
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            return context
        except Exception as e:
            st.error(f"فشل في استرجاع السياق: {e}")
            return ""
    return ""

# --- دالة استعلام النموذج باستخدام مكتبة ollama (جديدة) ---
def query_ollama_model(prompt, model_name="llama3.1:latest"):
    """Query the Ollama model using the official library."""
    try:
        response = ollama.chat(model=model_name, messages=[
            {'role': 'user', 'content': prompt}
        ])
        return response['message']['content']
    except Exception as e:
        st.error(f"خطأ أثناء استدعاء نموذج Ollama: {e}")
        # محاولة تقديم تفاصيل أكثر إذا كان الخطأ متعلق بالاتصال
        if "Connection refused" in str(e) or "Failed to connect" in str(e):
             st.error("تأكد من أن خادم Ollama يعمل في الخلفية.")
        return "حدث خطأ أثناء معالجة الطلب من النموذج."

# --- تحديث دالة get_conversational_chain ---
def get_answer_from_model(context, question):
    """Create the prompt and query the model."""
    # استخدام قالب أبسط أو تعديله حسب الحاجة
    # يمكنك تجربة قوالب مختلفة لمعرفة الأفضل لـ Llama 3.1
    prompt_template = f"""
استنادًا إلى السياق التالي، أجب عن السؤال بأكبر قدر ممكن من الدقة والتفصيل.
إذا كانت الإجابة غير موجودة في السياق، فقل ذلك بوضوح.

السياق:
{context}

السؤال:
{question}

الإجابة:
"""
    # استدعاء النموذج باستخدام الدالة الجديدة
    response = query_ollama_model(prompt_template)
    return response

# --- تحديث دالة user_input ---
def user_input(user_question):
    """Processes user input, retrieves context, and calls the model."""
    vector_store = st.session_state.get("vector_store") # الحصول على تخزين المتجهات من حالة الجلسة
    if not vector_store:
        st.error("لم يتم تحميل تخزين المتجهات. يرجى تحميل ملف PDF أولاً.")
        return

    if user_question:
        with st.spinner("جاري البحث عن السياق واستدعاء النموذج..."):
            # 1. استرجاع السياق ذي الصلة
            context = get_relevant_context(vector_store, user_question)
            if not context:
                st.warning("لم يتم العثور على سياق ذي صلة في المستندات.")
                # يمكنك اختيار إيقاف هنا أو محاولة الاستعلام بدون سياق محدد
                # response = query_ollama_model(f"السؤال: {user_question}\nالإجابة:")

            # 2. استدعاء النموذج بالسياق ذي الصلة
            response = get_answer_from_model(context, user_question)

            # 3. عرض الإجابة
            st.write("الرد:", response)
    else:
        st.warning("الرجاء إدخال سؤال.")

# --- تحديث الدالة الرئيسية لاستخدام حالة الجلسة ---
def main():
    """Main function of the Streamlit application."""
    st.set_page_config(page_title="CHAT WITH YOUR PDF")
    st.header("تحدث مع ملفات PDF الخاصة بك")

    # استخدام حالة الجلسة لتخزين تخزين المتجهات وتجنب إعادة المعالجة
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []

    with st.sidebar:
        st.subheader("ملفات PDF")
        pdf_docs = st.file_uploader("قم بتحميل ملفات PDF هنا واضغط على 'معالجة'", accept_multiple_files=True, type="pdf")

        # زر للمعالجة لتجنب إعادة المعالجة عند كل تغيير بسيط
        if st.button("معالجة الملفات"):
            if pdf_docs:
                # تحقق مما إذا كانت الملفات قد تغيرت
                current_file_names = sorted([pdf.name for pdf in pdf_docs])
                if current_file_names != st.session_state.processed_files:
                    with st.spinner("جاري قراءة ومعالجة ملفات PDF..."):
                        raw_text = pdf_read(pdf_docs)
                        if raw_text:
                            text_chunks = create_text_chunks(raw_text)
                            if text_chunks:
                                st.session_state.vector_store = create_and_save_vector_store(text_chunks)
                                st.session_state.processed_files = current_file_names # تحديث قائمة الملفات المعالجة
                            else:
                                st.error("لم يتم العثور على نص في ملفات PDF أو فشل التقسيم.")
                                st.session_state.vector_store = None
                                st.session_state.processed_files = []
                        else:
                            st.error("فشل في قراءة النص من ملفات PDF.")
                            st.session_state.vector_store = None
                            st.session_state.processed_files = []
                else:
                    st.info("تمت معالجة هذه الملفات بالفعل.")
            else:
                st.warning("الرجاء تحميل ملف PDF واحد على الأقل.")
                st.session_state.vector_store = None
                st.session_state.processed_files = []

        st.subheader("مراقبة الموارد")
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / (1024 ** 2)  # Conversion to megabytes
        st.write(f"استخدام الذاكرة: {memory_usage:.2f} MB")

        # زر اختياري لتحميل تخزين متجهات موجود مسبقًا
        if st.button("تحميل تخزين متجهات موجود"):
             st.session_state.vector_store = load_vector_store()


    st.subheader("اطرح سؤالاً")
    user_question = st.text_input("أدخل سؤالك حول محتوى ملفات PDF:")

    # عرض الإجابة فقط إذا تم تحميل الملفات وطرح سؤال
    if st.session_state.vector_store and user_question:
         user_input(user_question)
    elif user_question and not st.session_state.vector_store:
        st.warning("يرجى تحميل ومعالجة ملفات PDF أولاً.")


if __name__ == "__main__":
    # تعيين متغير البيئة (على الرغم من أنه قد لا يكون ضروريًا مع SpacyEmbeddings)
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    main()