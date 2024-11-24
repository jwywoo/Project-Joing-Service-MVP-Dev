import torch
from torch import nn
from rec_system.method.gmf import GMF
from rec_system.method.mlp import MLP
from rec_system.method.engine import Engine
from rec_system.method.utils import use_cpu, resume_checkpoint


class NeuMF(torch.nn.Module):
    def __init__(self, config):
        super(NeuMF, self).__init__()
        self.config = config
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim_mf = config['latent_dim_mf']
        self.latent_dim_mlp = config['latent_dim_mlp']
        self.meta_latent_dim = config['meta_latent_dim']

        # MLP 임베딩 레이어
        self.embedding_user_mlp = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim_mlp)
        self.embedding_item_mlp = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim_mlp)

        # GMF 임베딩 레이어
        self.embedding_user_mf = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim_mf)
        self.embedding_item_mf = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim_mf)

        # MLP 모델의 추가적인 메타데이터 임베딩
        self.embedding_item_category = torch.nn.Embedding(config['num_item_categories'], self.meta_latent_dim)
        self.embedding_media_type = torch.nn.Embedding(2, self.meta_latent_dim)  # 2: short, long
        self.embedding_channel_category = torch.nn.Embedding(config['num_channel_categories'], self.meta_latent_dim)
        self.embedding_subscribers = torch.nn.Embedding(config['max_subscribers'], self.meta_latent_dim)

        # MLP의 FC 레이어
        self.fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(config['layers'][:-1], config['layers'][1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        self.affine_output = torch.nn.Linear(in_features=config['layers'][-1] + config['latent_dim_mf'], out_features=1)
        self.logistic = torch.nn.Sigmoid()

        # Gaussian 초기화
        if config['weight_init_gaussian']:
            for sm in self.modules():
                if isinstance(sm, (nn.Embedding, nn.Linear)):
                    torch.nn.init.normal_(sm.weight.data, 0.0, 0.01)

    def forward(self, user_indices, item_indices, item_category, media_type, channel_category, subscribers):
        # MLP 임베딩
        user_embedding_mlp = self.embedding_user_mlp(user_indices)
        item_embedding_mlp = self.embedding_item_mlp(item_indices)

        # GMF 임베딩
        user_embedding_mf = self.embedding_user_mf(user_indices)
        item_embedding_mf = self.embedding_item_mf(item_indices)

        # 메타데이터 임베딩
        item_category_embedding = self.embedding_item_category(item_category)
        media_type_embedding = self.embedding_media_type(media_type)
        channel_category_embedding = self.embedding_channel_category(channel_category)
        subscribers_embedding = self.embedding_subscribers(subscribers)

        # MLP 벡터
        mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp, item_category_embedding, media_type_embedding,
                               channel_category_embedding, subscribers_embedding], dim=-1)

        # GMF 벡터 (내적)
        mf_vector = torch.mul(user_embedding_mf, item_embedding_mf)

        # MLP 레이어
        for idx in range(len(self.fc_layers)):
            mlp_vector = self.fc_layers[idx](mlp_vector)
            mlp_vector = torch.nn.ReLU()(mlp_vector)

        # MLP와 GMF 벡터 결합
        vector = torch.cat([mlp_vector, mf_vector], dim=-1)
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating

    def init_weight(self):
        pass

    def load_pretrain_weights(self):
        """Loading weights from trained MLP model & GMF model"""
        config = self.config
        config['latent_dim'] = config['latent_dim_mlp']
        mlp_model = MLP(config)
        if config['use_cuda'] is True:
            mlp_model.cuda()
        resume_checkpoint(mlp_model, model_dir=config['pretrain_mlp'], device_id=config['device_id'])

        self.embedding_user_mlp.weight.data = mlp_model.embedding_user.weight.data
        self.embedding_item_mlp.weight.data = mlp_model.embedding_item.weight.data
        for idx in range(len(self.fc_layers)):
            self.fc_layers[idx].weight.data = mlp_model.fc_layers[idx].weight.data

        config['latent_dim'] = config['latent_dim_mf']
        gmf_model = GMF(config)
        if config['use_cuda'] is True:
            gmf_model.cuda()
        resume_checkpoint(gmf_model, model_dir=config['pretrain_mf'], device_id=config['device_id'])
        self.embedding_user_mf.weight.data = gmf_model.embedding_user.weight.data
        self.embedding_item_mf.weight.data = gmf_model.embedding_item.weight.data

        self.affine_output.weight.data = 0.5 * torch.cat([mlp_model.affine_output.weight.data, gmf_model.affine_output.weight.data], dim=-1)
        self.affine_output.bias.data = 0.5 * (mlp_model.affine_output.bias.data + gmf_model.affine_output.bias.data)


class NeuMFEngine(Engine):
    """Engine for training & evaluating NeuMF model"""
    def __init__(self, config):
        self.model = NeuMF(config)
        self.model.to(use_cpu())
        super(NeuMFEngine, self).__init__(config)
        print(self.model)

        if config['pretrain']:
            self.model.load_pretrain_weights()

    def train_an_epoch_neumf(self, train_loader, epoch_id):
        self.model.train()
        total_loss = 0
        print(f"Starting training for epoch {epoch_id}...")
        for batch_id, batch in enumerate(train_loader):
            user = batch['user_id']
            item = batch['item_id']
            rating = batch['target'].float()
            item_category = batch['item_category']
            media_type = batch['media_type']
            channel_category = batch['channel_category']
            subscribers = batch['subscribers']

            loss = self.train_single_batch_neumf(user, item, rating, item_category, media_type, channel_category,
                                                 subscribers)
            print(f'[Training Epoch {epoch_id}] Batch {batch_id}, Loss {loss}')
            total_loss += loss


        self._writer.add_scalar('model/loss', total_loss, epoch_id)

    def train_single_batch_neumf(self, users, items, ratings, item_category, media_type, channel_category, subscribers):
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