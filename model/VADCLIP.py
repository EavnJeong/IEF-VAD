import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from .module import QuickGELU, Transformer
from .layers import GraphConvolution, DistanceAdj
from .clip import clip


class CLIPVAD(nn.Module):
    def __init__(self,
            num_class: int,
            embed_dim: int,
            visual_length: int,
            visual_width: int,
            visual_head: int,
            visual_layers: int,
            attn_window: int,
            prompt_prefix: int,
            prompt_postfix: int,
            device,
            args
        ):
        super().__init__()

        self.num_class = num_class
        self.visual_length = visual_length
        self.visual_width = visual_width
        self.embed_dim = embed_dim
        self.attn_window = attn_window
        self.prompt_prefix = prompt_prefix
        self.prompt_postfix = prompt_postfix
        self.device = device

        self.temporal = Transformer(
            width=visual_width,
            layers=visual_layers,
            heads=visual_head,
            attn_mask=self.build_attention_mask(self.attn_window)
        )

        width = int(visual_width / 2)
        self.gc1 = GraphConvolution(visual_width, width, residual=True)
        self.gc2 = GraphConvolution(width, width, residual=True)
        self.gc3 = GraphConvolution(visual_width, width, residual=True)
        self.gc4 = GraphConvolution(width, width, residual=True)
        self.disAdj = DistanceAdj()
        self.linear = nn.Linear(visual_width, visual_width)
        self.gelu = QuickGELU()

        self.mlp1 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 4, visual_width))
        ]))
        self.mlp2 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 4, visual_width))
        ]))
        self.classifier = nn.Linear(visual_width, 1)

        if 'vitb' in args.ds:
            self.clipmodel, _ = clip.load("ViT-B/32", device)
        elif 'vitl' in args.ds:
            self.clipmodel, _ = clip.load("ViT-L/14", device)

        for clip_param in self.clipmodel.parameters():
            clip_param.requires_grad = False

        self.frame_position_embeddings = nn.Embedding(visual_length, visual_width)
        self.text_prompt_embeddings = nn.Embedding(77, self.embed_dim)

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.text_prompt_embeddings.weight, std=0.01)
        nn.init.normal_(self.frame_position_embeddings.weight, std=0.01)

    def build_attention_mask(self, attn_window):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.visual_length, self.visual_length)
        mask.fill_(float('-inf'))
        for i in range(int(self.visual_length / attn_window)):
            if (i + 1) * attn_window < self.visual_length:
                mask[i * attn_window: (i + 1) * attn_window, i * attn_window: (i + 1) * attn_window] = 0
            else:
                mask[i * attn_window: self.visual_length, i * attn_window: self.visual_length] = 0

        return mask

    def adj4(self, x, seq_len):
        soft = nn.Softmax(1)
        x2 = x.matmul(x.permute(0, 2, 1)) # B*T*T
        x_norm = torch.norm(x, p=2, dim=2, keepdim=True)  # B*T*1
        x_norm_x = x_norm.matmul(x_norm.permute(0, 2, 1))
        x2 = x2/(x_norm_x+1e-20)
        output = torch.zeros_like(x2)
        if seq_len is None:
            for i in range(x.shape[0]):
                tmp = x2[i]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i] = adj2
        else:
            for i in range(len(seq_len)):
                tmp = x2[i, :seq_len[i], :seq_len[i]]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i, :seq_len[i], :seq_len[i]] = adj2

        return output

    def encode_video(self, images, padding_mask, lengths):
        # 입력 비디오 데이터를 float 형으로 변환
        images = images.to(torch.float)

        # 위치 임베딩 계산을 위한 위치 ID 생성 (각 프레임에 대해 고유한 ID 부여)
        position_ids = torch.arange(self.visual_length, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand(images.shape[0], -1)

        # 위치 임베딩을 각 프레임에 매핑
        frame_position_embeddings = self.frame_position_embeddings(position_ids)
        frame_position_embeddings = frame_position_embeddings.permute(1, 0, 2)

        # 비디오 프레임 특징에 위치 임베딩 추가
        images = images.permute(1, 0, 2) + frame_position_embeddings

        # Transformer를 통해 시간적 연관성을 고려하여 비디오 특징 인코딩
        x, _ = self.temporal((images, None))
        x = x.permute(1, 0, 2)

        # 그래프 인접 행렬 (adj) 생성
        adj = self.adj4(x, lengths)
        # 거리 기반 인접 행렬 생성 (disadj)
        disadj = self.disAdj(x.shape[0], x.shape[1])

        # 첫 번째 그래프 컨볼루션 적용 (adj 기반)
        x1_h = self.gelu(self.gc1(x, adj))
        # 두 번째 그래프 컨볼루션 적용 (disadj 기반)
        x2_h = self.gelu(self.gc3(x, disadj))

        # 두 번째 그래프 컨볼루션 적용 후 추가 활성화 함수 (adj 기반)
        x1 = self.gelu(self.gc2(x1_h, adj))
        # 두 번째 그래프 컨볼루션 적용 후 추가 활성화 함수 (disadj 기반)
        x2 = self.gelu(self.gc4(x2_h, disadj))

        # 두 가지 그래프 결과를 합치고, 선형 변환 적용
        x = torch.cat((x1, x2), 2)
        x = self.linear(x)

        return x  # 최종 비디오 특징 반환

    def encode_textprompt(self, text):
        # 텍스트 데이터를 CLIP의 토큰화 함수로 처리
        word_tokens = clip.tokenize(text).to(self.device)

        # 토큰화된 텍스트 데이터를 CLIP 모델로 임베딩
        word_embedding = self.clipmodel.encode_token(word_tokens)

        # 텍스트 임베딩 초기화 및 반복 개수에 맞게 확장
        text_embeddings = self.text_prompt_embeddings(torch.arange(77).to(self.device)).unsqueeze(0).repeat([len(text), 1, 1])
        text_tokens = torch.zeros(len(text), 77).to(self.device)

        for i in range(len(text)):
            # 토큰 길이에 따라 적절히 텍스트 임베딩 값 재구성
            ind = torch.argmax(word_tokens[i], -1)
            text_embeddings[i, 0] = word_embedding[i, 0]
            text_embeddings[i, self.prompt_prefix + 1: self.prompt_prefix + ind] = word_embedding[i, 1: ind]
            text_embeddings[i, self.prompt_prefix + ind + self.prompt_postfix] = word_embedding[i, ind]
            text_tokens[i, self.prompt_prefix + ind + self.prompt_postfix] = word_tokens[i, ind]

        # CLIP 텍스트 인코더를 사용하여 최종 텍스트 특징 계산
        text_features = self.clipmodel.encode_text(text_embeddings, text_tokens)

        return text_features  # 최종 텍스트 특징 반환

    def forward(self, visual, padding_mask, text, lengths):
        # 비디오 데이터를 인코딩하여 시각적 특징 추출
        visual_features = self.encode_video(visual, padding_mask, lengths)

        # 비디오 특징 기반으로 분류 로짓 계산
        logits1 = self.classifier(visual_features + self.mlp2(visual_features))

        # 텍스트 데이터 인코딩
        text_features_ori = self.encode_textprompt(text)
        text_features = text_features_ori

        # 비디오 특징과 분류 로짓 간 상호작용 계산
        logits_attn = logits1.permute(0, 2, 1)  # logits1의 형태 변환
        visual_attn = logits_attn @ visual_features  # 비디오와 로짓 간 내적
        visual_attn = visual_attn / visual_attn.norm(dim=-1, keepdim=True)  # 정규화
        visual_attn = visual_attn.expand(visual_attn.shape[0], text_features_ori.shape[0], visual_attn.shape[2])  # 크기 확장

        # 텍스트 특징과 비디오 특징 결합
        text_features = text_features_ori.unsqueeze(0)  # 차원 확장
        text_features = text_features.expand(visual_attn.shape[0], text_features.shape[1], text_features.shape[2])  # 크기 확장
        text_features = text_features + visual_attn  # 결합
        text_features = text_features + self.mlp1(text_features)  # MLP 적용 후 결합

        # 비디오와 텍스트 특징 정규화
        visual_features_norm = visual_features / visual_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features_norm.permute(0, 2, 1)

        # 비디오-텍스트 특징 간 유사도 로짓 계산
        logits2 = visual_features_norm @ text_features_norm.type(visual_features_norm.dtype) / 0.07

        return text_features_ori, logits1, logits2  # 텍스트 특징, 비디오 로짓, 유사도 로짓 반환