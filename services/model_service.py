import torch
from torchvision import transforms
from models.custom_model import Model
import yaml

# 모델 로드 설정
model_path = 'custom.pth'
yaml_path = 'custom.yaml'

# YAML 설정 파일 로드
with open(yaml_path, 'r') as f:
    config = yaml.safe_load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
custom_model = Model(config['network_params']['input_channel'],
                     config['network_params']['output_channel'],
                     config['network_params']['hidden_size'],
                     len(config['character_list']),
                     config['network_params']['num_fiducial'],
                     config['imgH'], config['imgW'],
                     config['network_params']['batch_max_length'])

# 모델 로드
checkpoint = torch.load(model_path, map_location=device)
custom_model.load_state_dict(checkpoint, strict=False)
custom_model.eval()

# 이미지 전처리 설정
transform = transforms.Compose([
    transforms.Resize((config['imgH'], config['imgW'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
