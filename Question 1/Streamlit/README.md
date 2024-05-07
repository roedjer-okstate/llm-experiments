# Question 1 QnA with Streamlit

## Team SFRR Analytics

NOTE: View the video in demo folder for the sample demonstration of the chatbot.

Please follow the steps below to run the chatbot:
1. Make sure the following files are in this directory:
- question1_qa_streamlit.py
- create_chat_history_csv.py
- Economy_of_China.pdf
- Economy_of_Germany.pdf
- Economy_of_India.pdf
- Economy_of_Japan.pdf
- Economy_of_the_United_Kingdom.pdf
- Economy_of_the_United_States.pdf
2. Make sure you have OPENAI_API_KEY in your os environment.
3. Open terminal in the same directory of `question1_qa_streamlit.py`.
4. Run `python create_chat_history_csv.py`. This creates/clears the chat history csv. Chat history csv has to be in static folder to run the next line.
5. Run `question1_qa_streamlit.py`. It may take some time to load the pdf into vector stores. Please be patient. Chatbot browser will appear and enjoy talking to the chatbot!
