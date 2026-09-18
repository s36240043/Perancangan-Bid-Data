import streamlit as st

st.title("Calculator Jovan Ganteng Uhuyyy")

result = None

num_1 = st.number_input("Masukan Number pertama :", value=0.0, step=1.0)
num_2 = st.number_input("Masukan Number kedua :", value=0.0, step=1.0)

st.subheader('Operations')
operation = st.radio('Select Operations :', ['+', '-', '*', '/'])

if st.button('Calculate Result'):
    if operation == '+':
        result = num_1 + num_2
    elif operation == '-':
        result = num_1 - num_2
    elif operation == '*':
        result = num_1 * num_2
    elif operation == '/':
        if num_2 == 0:
            st.warning('Tidak bisa di bagi dengan 0')
            result = None
        else:
            result = num_1 / num_2

    if result is not None:
        st.success(f"Hasil : {result}")