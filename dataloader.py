from torch.utils.data import Dataset

# take all the 3D images, and sort them under each modality
# i.e. {CT:[ ], CBV:[], ...}

class ISLES2018_loader(Dataset):
    def __init__(self, directory, modalities=None):
        super().__init__()

        self.data = {}

        for modality in modalities:
            self.data.update({modality : []})
    
    def viewData(self):
        print(self.data)






modalities = ['OT', 'CT', 'CT_CBV', 'CT_CBF', 'CT_Tmax' , 'CT_MTT']
dataset = ISLES2018_loader(None, modalities)
dataset.viewData()

