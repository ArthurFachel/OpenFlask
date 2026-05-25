# Tutorial: OpenClaw no Amazon Lightsail

## Pré-requisitos

- Conta na AWS com acesso ao Amazon Lightsail
- Chave de API do Gemini (Gratuito)
- Chave de API do Tavily (Gratuito)

---

## Passo 1 — Criar a instância no Lightsail

1. Acesse o [Amazon Lightsail](https://lightsail.aws.amazon.com)
2. Crie uma nova instância selecionando o blueprint **OpenClaw**

---

## Passo 2 — Configurar o Firewall

1. Na instância criada, acesse a aba **Networking**
2. Em **IPv4 Firewall**, clique em **Add rule**
3. Configure a regra:
   - **Protocol:** All protocols
   - **Source:** Any IPv4 address
4. Salve a regra

---

## Passo 3 — Configurar o OpenClaw via SSH

1. Conecte-se à instância via SSH
2. Execute o comando de onboarding:

```bash
openclaw onboard
```

3. Siga as instruções para configurar:
   - O **modelo** desejado
   - Sua **API Key**

---

## Passo 4 — Verificar o funcionamento

Execute o diagnóstico para checar o status das portas:

```bash
openclaw doctor
```

Caso algum problema seja identificado, execute a correção automática:

```bash
openclaw doctor --fix
```

---
## Passo 5 - Iniciar o Gateway 
```bash
   openclaw gateway 
```

---
## Passo 6 - Instalar dependencias 
```bash
   sudo apt update 
   sudo apt install python3
   sudo apt install python3-flask
   sudo apt install python3-pip
   pip install tavily-python --break-system-packages 
   pip install dotenv --break-system-packages
   pip install flask-limiter --break-system-packages
   pip install sentence-transformers numpy pdfplumber --break-system-packages
```
---
## Passo 7 - Resgatar sua API KEY do site do Tavily 

Acesse o site do [Tavily](https://app.tavily.com/home) para liberar a WebSearch

---
## Passo 8 - Criar um arquivo .env

Adicionar sua api key como:

```bash
TAVILY_API_KEY=
```

---
## Passo 9  — Adicionar o arquivo app.py via nano 

```bash
  nano app.py 
```

---

## Passo 10 — Passe o arquivo para RAG por SCP

Na aba do Lightsail, em Connect, baixe a chave SSH

Abra um terminal no seu computador e digite 

```bash
  chmod 400 path_para_a_chavessh path_para_o_documento
```
```bash
   scp -i path_para_a_chavessh.pem path_para_o_documento   ubuntu@SEU_IP_LIGHTSAIL:~/docs
```

## Resultado

### Teste a API em um jupyter notebook no VSCODE 

```bash 
import requests

url = "http://IPV4_DA_MAQUINA:5000/api/message"
headers = {
    "Content-Type": "application/json"
}
payload = {"query": "", #pergunta
           "inner_context": "", #contexto especifico
           "outer_context": "", #contexto abrangente 
            "search": False,    #pesquisa na web
            "rag": True}        #rag de documentos
response = requests.post(url, json=payload, headers=headers, timeout=120, stream=True)
print(response.text)
```
