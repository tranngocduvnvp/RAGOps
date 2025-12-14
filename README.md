# Deploy RAGOps
make secret
```
kubectl create secret generic ragapp-secret \
  --from-literal=GEMINI_API_KEY=your_gemini_api_key_here

```
# Deploy Open WebUI 
```
helm repo add open-webui https://helm.openwebui.com/
helm repo update
# download to modify 
helm search repo open-webui
helm pull open-webui/open-webui --untar
helm upgrade --install open-webui open-webui/open-webui
```