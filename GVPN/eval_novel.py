# Name: Jiaming Yu
# Time:
from model import *
import numpy as np
import time
def eval(datapath, model_path, num_classes):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    test_data = np.genfromtxt(datapath).reshape(1, 1, 32, 32, 32)
    test_data = torch.from_numpy(test_data).to(torch.float32)
    '''label_list = np.genfromtxt(label_path, dtype=np.int32).tolist()
    label = np.zeros(num_classes)
    label[label_list] = 1'''

    model = MyNBVNetV2(num_classes=num_classes)
    #model = ResNeXt(layers=[3, 4, 23, 3], cardinality=32, num_classes=num_classes)
    model = model.to(device)

    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])

    print('EVALUATING')
    model.eval()
    grid = test_data.to(device)

    startTime = time.time()

    output = model(grid)

    endTime = time.time()
    print('run time is ' + str(endTime - startTime))
    #print('output:', output[0])
    output[output >= 0.3] = 1
    output[output < 0.3] = 0
    '''correct1 = 0
    wrong1 = 0
    cnt1 = 0
    recall = 0
    precision = 0
    for j in range(num_classes):
        if label[j] == 1 and output[0][j] == 1:
            correct1 += 1
            cnt1 += 1
        elif label[j] == 1 and output[0][j] == 0:
            cnt1 += 1
        elif label[j] == 0 and output[0][j] == 1:
            wrong1 += 1
    recall += (correct1 / cnt1)
    precision += (correct1 / (correct1 + wrong1 + 1e-6))
    print(output.shape)
    print('recall:', recall, 'precision:', precision)'''
    #visualize_features_vertical(model.flat_feature_map, model.final_output, label)

    return output

if __name__ == "__main__":
    '''model = MyNBVNetV6().to('cuda:0')
    # block = RVPBlock(64, 128, residual=True)
    x = torch.randn(64, 1, 32, 32, 32).to('cuda:0')
    print(model(x).shape)
    # # print(model) '''
    num_classes = 40
    name_of_models = ['012', '032', '054', '062', '066', '077', '081', '085', '089', '099']
    for name_of_model in name_of_models:
        ''' t = 0
        r = 0
        v = 0'''
        # label_path = 'Boneviewsid000.txt'
        #model_path = 'D:/Programfiles/Myscvp/SCVPNet/trained_model/mydata40_ResNeXt_101_Adam_MyLoss_la0_1_la1_1_e150_b32_l0.0008520272550880615_gamma0.3classes40.pth.tar'
        model_path = 'D:/Programfiles/Myscvp/SCVPNet/trained_model/mydata40_NBVNet2_Adam_pro_MyLoss_pro0.75_e150_b64_l0.00015256689053291735_gamma0.3classes40.pth.tar'

        '''pred = eval(
            f'D:/Programfiles/Myscvp/industrial_label_data/test/40_views/novel/{name_of_model}/toward{t}_rotate{r}_view{v}/grid_toward{t}_rotate{r}_view{v}.txt',
            f'D:/Programfiles/Myscvp/industrial_label_data/test/40_views/novel/{name_of_model}/toward{t}_rotate{r}_view{v}/ids_toward{t}_rotate{r}_view{v}.txt',
            model_path, num_classes)'''
        pred = eval(f'D:/Programfiles/Myscvp/industrial_label_data/test/40_views/novel/{name_of_model}.txt',model_path, num_classes)
        ans = []
        for i in range(pred.shape[1]):
            if pred[0][i] == 1:
                print(i)
                ans.append(i)
        print('ans:', ans)
        np.savetxt('./log/' + name_of_model + '_NBVNet.txt', ans, fmt='%d')
    print('All tests of objects finished.')