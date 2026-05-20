# Tutorial: OpenClaw no Amazon Lightsail

## Pré-requisitos

- Conta na AWS com acesso ao Amazon Lightsail
- Chave de API do Gemini

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
   apt install python3
   apt install flask
```
## Passo 7 — Adicionar o arquivo app.py via nano 

```bash
  nano app.py 
```
## Resultado

Com o gateway aberto, o Flask passa a aceitar conexões de outros computadores na rede. A instância está pronta para uso.
