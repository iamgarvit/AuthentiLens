import re

with open('/Users/garvit/Documents/Garvit/codes/cv_project/backend/inference.py', 'r') as f:
    text = f.read()

replacement = """class CustomDeepLabV3(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.model = smp.DeepLabV3Plus(
            encoder_name="resnet50",
            encoder_weights=None,
            in_channels=3,
            classes=num_classes,
        )
        self.model.decoder.block = nn.Sequential(
            nn.Dropout2d(p=0.3),
            self.model.decoder.block2,
        )
       
    def forward(self, x):
        return self.model(x)"""

text = re.sub(r'class CustomDeepLabV3\(nn\.Module\):.*?def forward\(self, x\):\n        return self\.model\(x\)', replacement, text, flags=re.DOTALL)

with open('/Users/garvit/Documents/Garvit/codes/cv_project/backend/inference.py', 'w') as f:
    f.write(text)
