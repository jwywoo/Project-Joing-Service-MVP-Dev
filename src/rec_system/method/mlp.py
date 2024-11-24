import torch
from rec_system.method.engine import Engine
from rec_system.method.utils import use_cpu
from torch import nn
from rec_system.method.gmf import GMF
from rec_system.method.utils import resume_checkpoint


class MLP(torch.nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.config = config
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']

        # 사용자 및 아이템 임베딩
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        # 메타데이터 임베딩
        self.embedding_item_category = torch.nn.Embedding(config['num_item_categories'], config['meta_latent_dim'])
        self.embedding_media_type = torch.nn.Embedding(2, config['meta_latent_dim'])  # 2: short, long
        self.embedding_channel_category = torch.nn.Embedding(config['num_channel_categories'],
                                                             config['meta_latent_dim'])
        self.embedding_subscribers = torch.nn.Embedding(config['max_subscribers'], config['meta_latent_dim'])

        # MLP의 FC 레이어
        self.fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(config['layers'][:-1], config['layers'][1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        # 출력 레이어
        self.affine_output = torch.nn.Linear(in_features=config['layers'][-1], out_features=1)
        self.logistic = torch.nn.Sigmoid()

        # 가중치 초기화 (Gaussian 분포)
        if config['weight_init_gaussian']:
            for sm in self.modules():
                if isinstance(sm, (nn.Embedding, nn.Linear)):
                    torch.nn.init.normal_(sm.weight.data, 0.0, 0.01)

    def forward(self, user_indices, item_indices, item_category, media_type, channel_category, subscribers):
        # 사용자 및 아이템 임베딩
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)

        # 메타데이터 임베딩
        item_category_embedding = self.embedding_item_category(item_category)
        media_type_embedding = self.embedding_media_type(media_type)
        channel_category_embedding = self.embedding_channel_category(channel_category)
        subscribers_embedding = self.embedding_subscribers(subscribers)

        # 모든 임베딩을 결합
        vector = torch.cat([user_embedding, item_embedding, item_category_embedding, media_type_embedding,
                            channel_category_embedding, subscribers_embedding], dim=-1)

        # MLP 레이어를 통한 계산
        for idx in range(len(self.fc_layers)):
            vector = self.fc_layers[idx](vector)
            vector = torch.nn.ReLU()(vector)

        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating

    def init_weight(self):
        pass

    def load_pretrain_weights(self):
        """Loading weights from trained GMF model"""
        config = self.config
        gmf_model = GMF(config)

        # Load GMF pre-trained weights if required
        if config['use_cpu'] is True:
            gmf_model.to(use_cpu())  # Move GMF model to CPU
        resume_checkpoint(gmf_model, model_dir=config['pretrain_mf'], device_id=config['device_id'])

        # Transfer GMF weights to MLP
        self.embedding_user.weight.data = gmf_model.embedding_user.weight.data
        self.embedding_item.weight.data = gmf_model.embedding_item.weight.data


class MLPEngine(Engine):
    """Engine for training & evaluating MLP model"""

    def __init__(self, config):
        self.model = MLP(config)

        # Use CPU instead of CUDA
        if config['use_cpu'] is True:
            self.model.to(use_cpu())  # Move the model to CPU
        super(MLPEngine, self).__init__(config)
        print(self.model)

        # If pretraining is required, load weights
        if config['pretrain']:
            self.model.load_pretrain_weights()

    def train_an_epoch_mlp(self, train_loader, epoch_id):
        self.model.train()
        total_loss = 0
        for batch_id, batch in enumerate(train_loader):
            user = batch['user_id']
            item = batch['item_id']
            rating = batch['target'].float()
            item_category = batch['item_category']
            media_type = batch['media_type']
            channel_category = batch['channel_category']
            subscribers = batch['subscribers']

            loss = self.train_single_batch_mlp(user, item, rating, item_category, media_type, channel_category,
                                               subscribers)
            print(f'[Training Epoch {epoch_id}] Batch {batch_id}, Loss {loss}')
            total_loss += loss

        self._writer.add_scalar('model/loss', total_loss, epoch_id)

    def train_single_batch_mlp(self, users, items, ratings, item_category, media_type, channel_category, subscribers):
        users, items, ratings = users.to(self.device), items.to(self.device), ratings.to(self.device)
        item_category, media_type, channel_category, subscribers = (
            item_category.to(self.device),
            media_type.to(self.device),
            channel_category.to(self.device),
            subscribers.to(self.device),
        )

        self.opt.zero_grad()
        ratings_pred = self.model(users, items, item_category, media_type, channel_category, subscribers)
        loss = self.crit(ratings_pred.view(-1), ratings)
        loss.backward()
        self.opt.step()
        return loss