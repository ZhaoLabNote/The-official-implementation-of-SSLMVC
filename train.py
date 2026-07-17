import torch
from network import Network
from metric import valid
import numpy as np
import argparse
import random
from loss import Loss
from dataloader import load_data
import os
import utils

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

Dataname = 'Caltech-5V'
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--temperature_f", default=0.5)
parser.add_argument("--temperature_l", default=0.5)
parser.add_argument("--learning_rate", default=0.0003)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument("--mse_epochs", default=200)
parser.add_argument("--con_epochs", default=50)
parser.add_argument("--feature_dim", default=512)
parser.add_argument("--high_feature_dim", default=128)
parser.add_argument("--k", default=1.2)
parser.add_argument("--beta", default=3)
parser.add_argument("--lamb", default=1)
parser.add_argument("--pretrain", default=True)
parser.add_argument("--T", default=5)
parser.add_argument("--fusion_dim", default=512, type=int)
parser.add_argument("--prfr", type=int, default=10, help="Print frequency")

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.dataset == "Caltech-2V":
    args.con_epochs = 100
    seed = 0
elif args.dataset == "Caltech-3V":
    args.con_epochs = 100
    seed = 0
elif args.dataset == "Caltech-4V":
    args.con_epochs = 100
    seed = 0
elif args.dataset == "Caltech-5V":
    args.con_epochs = 100
    seed = 5
else:
    seed = 0
    args.con_epochs = 100

def setup_seed(seed, deterministic=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

setup_seed(seed, deterministic=True)

dataset, dims, view, data_size, class_num = load_data(args.dataset)

data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
)

def contrastive_train(epoch, pseudo_labels=None, view_weights=None):
    tot_loss = 0.
    mes = torch.nn.MSELoss()

    for batch_idx, (xs, _, indices) in enumerate(data_loader):
        if pseudo_labels is not None:
            batch_indices = indices.cpu().numpy()

        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()

        hs, qs, xrs, zs, ps, fusion_h, fusion_q, fusion_xrs, fusion_p = model.forward(xs)
        loss_list = []

        fusion_recon_loss = 0
        for v in range(view):
            fusion_recon_loss += mes(xs[v], fusion_xrs[v])
        loss_list.append(fusion_recon_loss)

        fusion_consistency_loss = 0
        for v in range(view):
            fusion_consistency_loss += (1 - torch.nn.functional.cosine_similarity(fusion_h, hs[v], dim=1).mean())
        loss_list.append(fusion_consistency_loss * args.lamb)

        for v in range(view):
            for w in range(v + 1, view):
                q1, q2 = utils.process_output(ps[v], ps[w])
                loss_con = -1 * (torch.mean(torch.sum(q1 * torch.log(qs[w]), dim=1)) + torch.mean(
                    torch.sum(q2 * torch.log(qs[v]), dim=1))) / 2
                loss_list.append(loss_con)

                q1_f, _ = utils.process_output(ps[v], fusion_p)
                _, q2_f = utils.process_output(fusion_p, ps[w])
                fusion_lp_loss = -1 * (torch.mean(torch.sum(q1_f * torch.log(fusion_q), dim=1)) +
                                       torch.mean(torch.sum(q2_f * torch.log(qs[v]), dim=1))) / 2
                loss_list.append(fusion_lp_loss)

                if view_weights is not None:
                    v_weight = view_weights[v] * view_weights[w]
                    loss_list.append(v_weight * criterion.forward_feature(hs[v], hs[w]))
                else:
                    loss_list.append(criterion.forward_feature(hs[v], hs[w]))

                loss_list.append(args.k * criterion.forward_feature(fusion_h, hs[v]))
                loss_list.append(args.k * criterion.forward_feature(fusion_h, hs[w]))

                loss_list.append(args.beta * criterion.forward_label(qs[v], qs[w]))

                loss_list.append(args.k * args.beta * criterion.forward_label(fusion_q, qs[v]))
                loss_list.append(args.k * args.beta * criterion.forward_label(fusion_q, qs[w]))

                loss_list.append(criterion.VIC_loss(hs[v], hs[w], args.high_feature_dim))
                loss_list.append(criterion.VIC_q_loss(qs[v], qs[w]))
                loss_list.append(criterion.VIC_loss(fusion_h, hs[v], args.high_feature_dim))
                loss_list.append(criterion.VIC_q_loss(fusion_q, qs[v]))

                loss_list.append(args.lamb * mes(zs[v], zs[w]))

                if pseudo_labels is not None:
                    batch_p_kmeans = pseudo_labels[batch_indices]
                    loss_list.append(criterion.simplified_pseudo_label_loss(
                        qs[v], batch_p_kmeans, view, temperature=0.5))
                    loss_list.append(args.k * criterion.simplified_pseudo_label_loss(
                        fusion_q, batch_p_kmeans, view, temperature=0.5))

            loss_list.append(mes(xs[v], xrs[v]))

        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()

    if args.prfr > 0 and epoch % args.prfr == 0:
        print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))
    return np.round(tot_loss / len(data_loader), 4)

def pretrain(epoch):
    tot_loss = 0.
    criterion = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        hs, qs, xrs, zs, ps, _, _, fusion_xrs, _ = model(xs)

        loss_list = []
        for v in range(view):
            loss_list.append(criterion(xs[v], xrs[v]))

        for v in range(view):
            loss_list.append(criterion(xs[v], fusion_xrs[v]))

        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()

    if epoch % 100 == 0 or epoch == 1:
        print('=' * 50)
        print(f'Pretrain epoch {epoch}/{args.mse_epochs} complete - loss: {tot_loss / len(data_loader):.6f}')
        print('=' * 50)

if not os.path.exists('./pretrain_models'):
    os.makedirs('./pretrain_models')

print("=" * 80)
print(f"Starting training on dataset: {args.dataset}")
print("=" * 80 + "\n")

T = args.T
for i in range(T):
    print(f"ROUND: {i + 1}")
    round_seed = seed + i * 1
    setup_seed(round_seed)
    print(f"Using random seed: {round_seed}")

    model = Network(view, dims, args.feature_dim, args.high_feature_dim, class_num, args.fusion_dim, device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = Loss(args.batch_size, class_num, args.temperature_f, args.temperature_l, device).to(device)
    epoch = 1
    current_epoch = args.mse_epochs + 1
    pseudo_labels = None
    view_weights = None
    update_interval = 10
    warmup_epochs = 15
    pca_dim = None
    best_acc = 0
    best_epoch_nmi = 0
    best_epoch_pur = 0
    best_epoch_ari = 0
    best_epoch = 0
    best_epoch_loss = 0

    if args.pretrain:
        while epoch <= args.mse_epochs:
            pretrain(epoch)
            epoch += 1
        print("Pretraining complete, saving model...")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, './pretrain_models/' + args.dataset + '.pth')
        print("Model saved.")
    else:
        print("Loading pretrained model...")
        checkpoint = torch.load('./pretrain_models/' + args.dataset + '.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = args.mse_epochs + 1
        print("Pretrained model loaded.")

    while current_epoch <= args.mse_epochs + args.con_epochs:
        use_kmeans = current_epoch >= (args.mse_epochs + warmup_epochs)
        enable_global_p = use_kmeans

        if enable_global_p and ((current_epoch - args.mse_epochs - warmup_epochs) % update_interval == 0 or pseudo_labels is None):
            pseudo_labels, view_weights = utils.generate_pseudo_labels(
                model, device, dataset, view, class_num,
                update_interval=update_interval,
                pca_dim=pca_dim,
                current_epoch=current_epoch
            )

        if enable_global_p:
            loss_ = contrastive_train(current_epoch, pseudo_labels, view_weights)
        else:
            loss_ = contrastive_train(current_epoch, None, None)

        acc, nmi, ari, pur = valid(model, device, dataset, view, data_size, class_num, eval_h=False)

        if acc > best_acc:
            best_acc = np.copy(acc)
            best_epoch_nmi = np.copy(nmi)
            best_epoch_ari = np.copy(ari)
            best_epoch_pur = np.copy(pur)
            best_epoch = current_epoch
            best_epoch_loss = loss_

        if args.prfr > 0 and current_epoch % args.prfr == 0:
            print(f'ACC = {acc:.4f} NMI = {nmi:.4f} ARI = {ari:.4f} PUR = {pur:.4f}')

        if current_epoch == args.mse_epochs + args.con_epochs:
            print('-' * 50)
            print(args.dataset)
            print('Best results: ACC, NMI, ARI, PUR, EPOCH, LOSS')
            print('{:.4f} {:.4f} {:.4f} {:.4f} {} {:.6f}'.format(best_acc, best_epoch_nmi, best_epoch_ari,
                                                               best_epoch_pur, best_epoch, best_epoch_loss))

        current_epoch += 1

if args.T > 0:
    print("\n" + "=" * 50)
    print(f"{args.T} runs complete! results:")
    print(f"ACC: {best_acc:.4f}")
    print(f"NMI: {best_epoch_nmi:.4f}")
    print(f"ARI: {best_epoch_ari:.4f}")
    print(f"PUR: {best_epoch_pur:.4f}")
    print("=" * 50)
