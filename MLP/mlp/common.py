import torch

def calculate_accuracy(model, test_loader):
    correct = 0
    total = 0

    # 关闭梯度计算，提高效率
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            # 获取预测结果（概率最高的类别）
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'测试集准确率: {accuracy:.2f}%')
    print(f'正确识别: {correct}/{total}')
    return accuracy