import torch
import numpy as np

from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer, IntervalStrategy
from models.LMs.model import BertClassifier, BertClaInfModel
from models.LMs.lm_utils import Dataset
from models.LMs.lm_utils import load_data
from models.LMs.lm_utils import init_path, time_logger


def compute_metrics(p):
    from sklearn.metrics import accuracy_score
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    return {"accuracy": accuracy}


class LMTrainer():
    def __init__(self, args):
        self.dataset_name = args.dataset_name
        self.seed = args.seed

        self.model_name = args.model_name
        self.feat_shrink = args.feat_shrink

        self.weight_decay = args.weight_decay
        self.dropout = args.dropout
        self.att_dropout = args.att_dropout
        self.cla_dropout = args.cla_dropout
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.warmup_epochs = args.warmup_epochs
        self.eval_patience = args.eval_patience
        self.grad_acc_steps = args.grad_acc_steps
        self.lr = args.learning_rate

        self.use_llm_str = "2" if args.use_llm else ""
        self.output_dir = f'lm_outputs/{self.dataset_name}{self.use_llm_str}/{self.model_name}-seed{self.seed}'
        self.ckpt_dir = f'saved_lm_models/{self.dataset_name}{self.use_llm_str}/{self.model_name}-seed{self.seed}'
        self.dataset_path = f'{args.code_path}/../DyLink_Datasets/{self.dataset_name}'

        # Preprocess data
        #todo: this function need to be rewritten
        data, num_classes, text = load_data(
            dataset_path=self.dataset_path, batch_size=self.batch_size, val_ratio=args.val_ratio, test_ratio=args.test_ratio,
            use_llm=args.use_llm, seed=self.seed)
        self.data = data
        self.num_nodes = data.y.size(0)
        self.n_labels = num_classes

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        X = tokenizer(text, padding=True, truncation=True, max_length=512)

        dataset = Dataset(X, data.y.tolist())
        self.inf_dataset = dataset

        self.train_dataset = torch.utils.data.Subset(
            dataset, self.data.train_mask.nonzero().squeeze().tolist())
        self.val_dataset = torch.utils.data.Subset(
            dataset, self.data.val_mask.nonzero().squeeze().tolist())
        self.test_dataset = torch.utils.data.Subset(
            dataset, self.data.test_mask.nonzero().squeeze().tolist())

        # Define pretrained tokenizer and model
        bert_model = AutoModel.from_pretrained(f'{args.code_path}/../LMs/{self.model_name}')
        self.model = BertClassifier(bert_model,
                                    n_labels=self.n_labels,
                                    feat_shrink=self.feat_shrink)

        # prev_ckpt = f'prt_lm/{self.dataset_name}/{self.model_name}.ckpt'
        # if self.use_gpt_str and os.path.exists(prev_ckpt):
        #     print("Initialize using previous ckpt...")
        #     self.model.load_state_dict(torch.load(prev_ckpt))

        self.model.config.dropout = self.dropout
        self.model.config.attention_dropout = self.att_dropout

        trainable_params = sum(p.numel()
                               for p in self.model.parameters() if p.requires_grad)
        print(f"\nNumber of parameters: {trainable_params}")

    @time_logger
    def train(self):
        # Define training parameters
        eq_batch_size = self.batch_size * 4
        train_steps = self.num_nodes // eq_batch_size + 1
        eval_steps = self.eval_patience // eq_batch_size
        warmup_steps = int(self.warmup_epochs * train_steps)

        # Define Trainer
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            do_train=True,
            do_eval=True,
            eval_steps=eval_steps,
            evaluation_strategy=IntervalStrategy.STEPS,
            save_steps=eval_steps,
            learning_rate=self.lr,
            weight_decay=self.weight_decay,
            save_total_limit=1,
            load_best_model_at_end=True,
            gradient_accumulation_steps=self.grad_acc_steps,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size*8,
            warmup_steps=warmup_steps,
            num_train_epochs=self.epochs,
            dataloader_num_workers=1,
            fp16=True,
            dataloader_drop_last=True,
        )
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=compute_metrics,
            # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        # Train pre-trained model
        self.trainer.train()
        torch.save(self.model.state_dict(), init_path(f"{self.ckpt_dir}.ckpt"))
        print(f'LM saved to {self.ckpt_dir}.ckpt')

    @time_logger
    @torch.no_grad()
    def eval_and_save(self):
        emb = np.memmap(init_path(f"{self.ckpt_dir}.emb"),
                        dtype=np.float16,
                        mode='w+',
                        shape=(self.num_nodes, self.feat_shrink if self.feat_shrink else 768))
        pred = np.memmap(init_path(f"{self.ckpt_dir}.pred"),
                         dtype=np.float16,
                         mode='w+',
                         shape=(self.num_nodes, self.n_labels))

        inf_model = BertClaInfModel(
            self.model, emb, pred, feat_shrink=self.feat_shrink)
        inf_model.eval()
        inference_args = TrainingArguments(
            output_dir=self.output_dir,
            do_train=False,
            do_predict=True,
            per_device_eval_batch_size=self.batch_size*8,
            dataloader_drop_last=False,
            dataloader_num_workers=1,
            fp16_full_eval=True,
        )

        trainer = Trainer(model=inf_model, args=inference_args)
        trainer.predict(self.inf_dataset)
        if "ogbn" in self.dataset_name:
            from ogb.nodeproppred import Evaluator
            _evaluator = Evaluator(name=self.dataset_name)
        else:
            from core.GNNs.gnn_utils import Evaluator
            _evaluator = Evaluator(name=self.dataset_name)

        def evaluator(preds, labels): return _evaluator.eval({
            "y_true": torch.tensor(labels).view(-1, 1),
            "y_pred": torch.tensor(preds).view(-1, 1),
        })["acc"]

        def eval(x): return evaluator(
            np.argmax(pred[x], -1), self.data.y[x])

        train_acc = eval(self.data.train_mask)
        val_acc = eval(self.data.val_mask)
        test_acc = eval(self.data.test_mask)
        print(
            f'[LM] TrainAcc: {train_acc:.4f}, ValAcc: {val_acc:.4f}, TestAcc: {test_acc:.4f}\n')
        return {'TrainAcc': train_acc, 'ValAcc': val_acc, 'TestAcc': test_acc}
