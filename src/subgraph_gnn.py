import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.metrics import mean_squared_error


# 子图选择器
class SubgraphSelector(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(SubgraphSelector, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = torch.nn.Linear(hidden_channels, 1)  # 输出每个节点的概率

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        p = torch.sigmoid(self.fc(x))  # 节点属于子图的概率
        return p


# 图嵌入网络
class GraphEmbedder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphEmbedder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = torch.nn.Linear(hidden_channels, out_channels)  # 图嵌入维度

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        h = global_mean_pool(x, batch)  # 图池化生成全局嵌入
        return h


# 总体模型
class SubgraphGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, embed_dim):
        super(SubgraphGNN, self).__init__()
        self.selector = SubgraphSelector(in_channels, hidden_channels)
        self.embedder = GraphEmbedder(in_channels, hidden_channels, embed_dim)

    def forward(self, x_H, edge_index_H, x_G, edge_index_G, batch_H, batch_G):
        # 预测子图节点概率
        p = self.selector(x_H, edge_index_H)

        # 用预测的概率选择子图节点
        x_F = x_H * p  # 简单乘法权重示例
        h_F = self.embedder(x_F, edge_index_H, batch_H)

        # 图 G 的嵌入
        h_G = self.embedder(x_G, edge_index_G, batch_G)

        return h_F, h_G, p


def train(model, data_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in data_loader:
        optimizer.zero_grad()

        # 提取数据
        x_H, edge_index_H, batch_H = data.x_H, data.edge_index_H, data.batch_H
        x_G, edge_index_G, batch_G = data.x_G, data.edge_index_G, data.batch_G
        target_distance = data.target_distance

        # 前向传播
        h_F, h_G, p = model(x_H, edge_index_H, x_G, edge_index_G, batch_H, batch_G)

        # 计算距离
        predicted_distance = torch.norm(h_F - h_G, dim=1).mean()
        loss = criterion(predicted_distance, target_distance)

        # 反向传播与优化
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)


def test(model, data_loader):
    model.eval()
    distances = []
    predictions = []
    with torch.no_grad():
        for data in data_loader:
            x_H, edge_index_H, batch_H = data.x_H, data.edge_index_H, data.batch_H
            x_G, edge_index_G, batch_G = data.x_G, data.edge_index_G, data.batch_G
            target_distance = data.target_distance

            h_F, h_G, p = model(x_H, edge_index_H, x_G, edge_index_G, batch_H, batch_G)
            predicted_distance = torch.norm(h_F - h_G, dim=1).mean()

            distances.append(target_distance.item())
            predictions.append(predicted_distance.item())

    return mean_squared_error(distances, predictions), distances, predictions


if __name__ == "__main__":
    # 加载数据集
    dataset = torch.load("graph_dataset.pt")
    train_loader = DataLoader(dataset[:80], batch_size=16, shuffle=True)
    test_loader = DataLoader(dataset[80:], batch_size=16)

    # 模型初始化
    in_channels = dataset[0].x_H.size(1)
    hidden_channels = 32
    embed_dim = 16
    model = SubgraphGNN(in_channels, hidden_channels, embed_dim)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    # 训练循环
    epochs = 50
    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, criterion)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}")

    # 测试模型
    mse, distances, predictions = test(model, test_loader)
    print(f"Test MSE: {mse:.4f}")
