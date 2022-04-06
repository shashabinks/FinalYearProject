import torch
import timm.models.swin_transformer as tm


if __name__ == "__main__":
    batch_size = 4
    num_classes = 1  # one hot
    initial_kernels = 32
    
    
    
    net = tm.SwinTransformer(32,4,5,1)
    
    # torch.save(net.state_dict(), 'model.pth')
    CT = torch.randn(batch_size, 5, 32, 32)    # Batchsize, modal, hight,

    print("Input:", CT.shape)
    if torch.cuda.is_available():
        net = net.cuda()
        CT = CT.cuda()
        torch.cuda.empty_cache()

    segmentation_prediction = net(CT)
    print("Output:",segmentation_prediction.shape)