import torch
import torchvision.models as models


#save
model = models.vgg16(weights='IMAGENET1K_V1')
torch.save(model.state_dict(), 'model_weights.pth')

#node

model = models.vgg16() # we do not specify weights, i.e. create untrained model
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()#model.eval()) để đảm bảo rằng không có dropout hoặc batch normalization nào được áp dụng khi chạy mô hình trên dữ liệu mới.


