from search import search_prompt

def main():
    try:
        chain = search_prompt()
    except Exception as exc:
        print(f"Não foi possível iniciar o chat. Verifique a configuração: {exc}")
        return
    
    print("Faça sua pergunta:")
    print('Digite "sair" para encerrar.')

    while True:
        try:
            question = input("\nPERGUNTA: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEncerrando chat.")
            break

        if not question:
            print("RESPOSTA: Informe uma pergunta para continuar.")
            continue

        if question.lower() in {"sair", "exit", "quit"}:
            print("Encerrando chat.")
            break

        try:
            answer = chain(question)
        except Exception as exc:
            print(f"RESPOSTA: Erro ao processar a pergunta: {exc}")
            continue

        print(f"RESPOSTA: {answer}")

if __name__ == "__main__":
    main()