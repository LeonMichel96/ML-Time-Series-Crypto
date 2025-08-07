import streamlit as st


st.set_page_config(page_title='Time Series Cripto', page_icon=':bar_chart:', layout='wide')

pages = {'General ✅':[st.Page('pages/1_🚀_Intro.py', default=True)],
         'Data Science 🔮':[st.Page('pages/2_📈_Predictor.py')]}

pg = st.navigation(pages)
pg.run()
