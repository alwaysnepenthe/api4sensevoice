### running code
```less
cd Audi_Voice_ Module
conda create -n Audi_Voice_ Module python=3.10
conda activate Audi_Voice_ Module
conda install -c conda-forge ffmpeg
pip install -r requirements.txt
```
### version 1.1
- VAD detection
- real-time streaming recognition
- speaker verification

### 
运行指南：
后端：本机部署，请先运行server_wss.py，待运行后，打开前端页面，即可进行语音识别。
前端：打开client_wss.html，即可进行语音识别。

如需部署至服务器，请暴露端口，并提供SSL证书，以保证安全。