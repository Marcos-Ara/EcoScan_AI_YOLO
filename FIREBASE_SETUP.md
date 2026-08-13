# EcoScan AI — Firebase Authentication

## 1. Firebase

Abra o Firebase Console e entre no projeto `ecoscan-ai-e961f`.

Em **Configurações do projeto > Seus apps**, confirme que existe um app Web.

Copie o objeto `firebaseConfig` para o arquivo `config.js` deste projeto.

## 2. Authentication

No Firebase:

- Authentication > Sign-in method > Email/Password: **Ativar**
- Authentication > Sign-in method > Google: **Ativar**

## 3. Domínio

Em Authentication > Settings > Authorized domains, adicione o domínio onde o site será publicado.

Exemplos:

- `localhost`
- seu domínio do GitHub Pages
- seu domínio personalizado

## 4. O que já funciona no código

- Login com e-mail e senha
- Cadastro com e-mail e senha
- Atualização do nome do usuário
- E-mail de verificação
- Recuperação de senha
- Login/cadastro com Google
- Persistência da sessão
- Logout
- Bloqueio do sistema para usuários não autenticados

## 5. Importante

As senhas não são salvas no Firestore pelo EcoScan AI. O Firebase Authentication é responsável pelas credenciais.

## 6. Teste

1. Abra o site.
2. Clique em Criar Conta.
3. Cadastre um e-mail real.
4. Abra o e-mail de confirmação.
5. Clique no link de verificação.
6. Volte ao EcoScan AI e clique em "Já verifiquei meu e-mail".
7. Depois teste o Google.
