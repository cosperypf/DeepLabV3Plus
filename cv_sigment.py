import argparse
from utils.base_model import BaseModel

class DeepLabV3(BaseModel):
    def __init__(self):
        super(DeepLabV3, self).__init__()
        self.model = None

    def train(self, args):
        print("---> function call: def train(self, args):")
        return None

    def save(self):
        print("---> function call: def save(self):")
        # self.save_model_pkl(model=self.model)
        pass

    def load(self):
        print("---> function call: def load(self):")
        # self.model = self.load_model_pkl()
        pass

    def predict(self, args):
        print("---> function call: def predict(self, args):")
        return None


def main():
    op = DeepLabV3()
    parse = argparse.ArgumentParser()
    parse.add_argument('--image_folder', type=str, default="") #JPG图像目录
    parse.add_argument('--semantic_segmentation_folder', type=str, default="") #PNG标注目录
    parse.add_argument('--list_folder', type=str, default="") #列表文件目录
    parse.add_argument('--predict_data_dir', type=str, default="") #预测数据目录
    parse.add_argument('--predict_result_path', type=str, default="") #预测结果输出目录
    parse.add_argument('--tb_logdir', type=str, default="") #模型目录


    parse.add_argument('--num_classes', type=int, default=0) #模型要预测的类别数
    parse.add_argument('--base_learning_rate', type=float, default=0.01) #学习率
    parse.add_argument('--num_steps', type=int, default=10000) #训练步数
    parse.add_argument('--batch_size', type=int, default=2) #批处理大小

    parse.add_argument('--checkpoint_path', type=str, default="") #预训练模型路径
    parse.add_argument('--visualize_result', type=str, default="True") #是否展示可视化输出
    parse.add_argument('--predict_input_size', type=int, default=0) #预测时输入尺寸
    parse.add_argument('--train_crop_size', type=str, default="") #训练时裁剪尺寸，半角逗号分割，如513,513

    args, _ = parse.parse_known_args()

    util.prepare(['torch', 'torchvision', 'numpy', 'pillow', 'scikit-learn', 'tqdm', 'matplotlib', 'visdom'])
    
    print("====> args: ",args)
    print("====> _: ",_)

    op.parse_args(args)
    op.run_deep(args)

if __name__ == '__main__':
    main()
