"""
# NES Attack (Bản Giải Thích Chi Tiết, Khớp Repo Gốc)

Tệp này được viết theo phong cách "tương tự" file nes_attack.py chia sẻ trên mạng
(tức có cấu trúc đối tượng/class), nhưng **match logic** với repo hiện tại:
- TensorFlow 1.x graph mode
- Inception v3 từ tools/inception_v3_imagenet.py
- 3 chế độ: query-limited, partial-information, label-only
- Line-search + momentum + epsilon decay

## Mục tiêu của tệp

1. Làm tài liệu kỹ thuật ngay trong code (markdown-style trong docstring/comment).
2. Giữ công thức và luồng tương đồng attacks.py đang chạy.
3. Dễ đọc hơn để học thuật toán NES black-box từ paper gốc.

## Công thức ước lượng gradient kiểu NES (ý tưởng chính)

Với nhiễu đối xứng $u$ và $-u$ quanh điểm $x$:

$$
\hat{g} = \frac{1}{\sigma} \mathbb{E}[L(x + \sigma u) u]
$$

Trong code hiện tại dùng cặp đối xứng để giảm phương sai:
- lấy noise_pos ~ N(0, I)
- ghép noise = [noise_pos, -noise_pos]
- lấy trung bình có trọng số bởi loss.

---

Lưu ý:
- Tệp này có thể chạy độc lập như script: python nes_attack_explained_vi.py ...
- Tuy nhiên đây là bản "giải thích + tham chiếu", không thay thế trực tiếp attacks.py.
"""

from __future__ import absolute_import, division, print_function

import argparse
import json
import os
import shutil
import time
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.python.client import device_lib

from tools.imagenet_labels import label_to_name
from tools.inception_v3_imagenet import model
from tools.logging_utils import log_output, render_frame
from tools.utils import get_image, image_of_class, one_hot, pseudorandom_target


IMAGENET_PATH = r"D:\repos\dataset\competitions\imagenet-object-localization-challenge\ILSVRC\Data\CLS-LOC"
NUM_LABELS = 1000
SIZE = 299


@dataclass
class NESConfig:
    """Cấu hình tấn công, map 1-1 với args của main.py."""

    samples_per_draw: int = 50
    batch_size: int = 50
    target_class: int = None
    orig_class: int = None
    sigma: float = 1e-3
    epsilon: float = 0.05
    img_path: str = None
    img_index: int = None
    out_dir: str = "query_limited/"
    log_iters: int = 1
    restore: str = None
    momentum: float = 0.9
    max_queries: int = 10000
    save_iters: int = 50
    plateau_drop: float = 2.0
    min_lr_ratio: int = 200
    plateau_length: int = 5
    gpus: int = None
    imagenet_path: str = None
    visualize: bool = False
    max_lr: float = 1e-2
    min_lr: float = 5e-5
    top_k: int = -1
    adv_thresh: float = -1.0
    label_only: bool = False
    zero_iters: int = 100
    label_only_sigma: float = 1e-3
    starting_eps: float = 1.0
    starting_delta_eps: float = 0.5
    min_delta_eps: float = 0.1
    conservative: int = 2


class NESAttackTF1:
    """
    # NES Attack theo repo gốc (TF1)

    Class này tách rõ các phần:
    - setup dữ liệu
    - setup graph/loss
    - ước lượng gradient
    - loop tối ưu

    Thiết kế gần với style class-based của nes_attack.py (PyTorch),
    nhưng nội dung khớp với attacks.py trong repo hiện tại.
    """

    def __init__(self, cfg, devices):
        self.cfg = cfg
        self.devices = devices

        self.initial_img = None
        self.orig_class = None
        self.target_class = None

        self.lower = None
        self.upper = None
        self.adv = None

        self.batch_per_device = None
        self.goal_epsilon = None
        self.epsilon = None
        self.delta_epsilon = None

        self.k = None
        self.label_only = None
        self.zero_iters = None
        self.is_targeted = None

        # TensorFlow handles
        self.sess = None
        self.x = None
        self.eval_logits = None
        self.eval_preds = None
        self.eval_percent_adv = None
        self.grad_estimate = None
        self.final_losses = None

        # Logging
        self.writer = None
        self.log_file = None
        self.loss_vs_queries = None
        self.loss_vs_steps = None
        self.lr_vs_queries = None
        self.lr_vs_steps = None
        self.empirical_loss = None
        self.lr_placeholder = None

        # Optional visualize graph
        self.render_feed = None
        self.render_logits = None

    def _setup_image_and_classes(self):
        """Chọn ảnh gốc và target class."""
        if self.cfg.img_path:
            self.initial_img = np.asarray(Image.open(self.cfg.img_path).resize((SIZE, SIZE)))
            self.orig_class = self.cfg.orig_class
            self.initial_img = self.initial_img.astype(np.float32) / 255.0
        else:
            x, y = get_image(self.cfg.img_index, IMAGENET_PATH)
            self.initial_img = x
            self.orig_class = y

        if self.cfg.target_class is None:
            self.target_class = pseudorandom_target(self.cfg.img_index, NUM_LABELS, self.orig_class)
            print("chose pseudorandom target class: %d" % self.target_class)
        else:
            self.target_class = self.cfg.target_class

    def _setup_attack_state(self):
        """Khởi tạo epsilon-ball, ảnh adv ban đầu và tham số mode-specific."""
        self.epsilon = self.cfg.epsilon
        self.goal_epsilon = self.cfg.epsilon

        self.lower = np.clip(self.initial_img - self.cfg.epsilon, 0.0, 1.0)
        self.upper = np.clip(self.initial_img + self.cfg.epsilon, 0.0, 1.0)

        if self.cfg.restore:
            self.adv = np.clip(np.load(self.cfg.restore), self.lower, self.upper)
        else:
            self.adv = self.initial_img.copy()

        self.batch_per_device = self.cfg.batch_size // len(self.devices)

        # Partial-information setup
        self.k = self.cfg.top_k
        if self.k > 0:
            if self.target_class == -1:
                raise ValueError("Partial-information attack is a targeted attack.")
            self.adv = image_of_class(self.target_class, IMAGENET_PATH)
            self.epsilon = self.cfg.starting_eps
            self.delta_epsilon = self.cfg.starting_delta_eps
        else:
            self.k = NUM_LABELS
            self.delta_epsilon = 0.0

        # Label-only setup
        self.label_only = self.cfg.label_only
        self.zero_iters = self.cfg.zero_iters

        self.is_targeted = 1 if self.target_class >= 0 else -1

    def _build_graph(self):
        """Xây TF graph: eval model, loss cho 3 mode, và gradient estimator."""
        self.sess = tf.InteractiveSession()
        self.x = tf.placeholder(tf.float32, self.initial_img.shape)
        self.eval_logits, self.eval_preds = model(self.sess, tf.expand_dims(self.x, 0))

        if self.target_class >= 0:
            self.eval_percent_adv = tf.equal(self.eval_preds[0], tf.constant(self.target_class, tf.int64))
        else:
            self.eval_percent_adv = tf.not_equal(self.eval_preds[0], tf.constant(self.orig_class, tf.int64))

        one_hot_vec = one_hot(self.target_class if self.target_class >= 0 else self.orig_class, NUM_LABELS)
        labels = np.repeat(np.expand_dims(one_hot_vec, axis=0), repeats=self.batch_per_device, axis=0)

        def standard_loss(eval_points, noise):
            logits, _ = model(self.sess, eval_points)
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            return losses, noise

        def label_only_loss(eval_points, noise):
            tiled_points = tf.tile(tf.expand_dims(eval_points, 0), [self.zero_iters, 1, 1, 1, 1])
            noised_eval_im = tiled_points + tf.random_uniform(
                tf.shape(tiled_points), minval=-1, maxval=1
            ) * self.cfg.label_only_sigma
            logits, _ = model(self.sess, tf.reshape(noised_eval_im, (-1,) + self.initial_img.shape))
            _, inds = tf.nn.top_k(logits, k=self.k)
            real_inds = tf.reshape(inds, (self.zero_iters, self.batch_per_device, -1))
            rank_range = tf.range(start=self.k, limit=0, delta=-1, dtype=tf.float32)
            tiled_rank_range = tf.tile(
                tf.reshape(rank_range, (1, 1, self.k)), [self.zero_iters, self.batch_per_device, 1]
            )
            batches_in = tf.where(
                tf.equal(real_inds, self.target_class),
                tiled_rank_range,
                tf.zeros(tf.shape(tiled_rank_range)),
            )
            return 1 - tf.reduce_mean(batches_in, [0, 2]), noise

        def partial_info_loss(eval_points, noise):
            logits, _ = model(self.sess, eval_points)
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            _, inds = tf.nn.top_k(logits, k=self.k)
            good_inds = tf.where(tf.equal(inds, tf.constant(self.target_class)))
            good_images = good_inds[:, 0]
            losses = tf.gather(losses, good_images)
            noise = tf.gather(noise, good_images)
            return losses, noise

        loss_fn = label_only_loss if self.label_only else (partial_info_loss if self.k < NUM_LABELS else standard_loss)

        grad_estimates = []
        final_losses = []
        for i, device in enumerate(self.devices):
            with tf.device(device):
                print("loading on gpu %d of %d" % (i + 1, len(self.devices)))
                noise_pos = tf.random_normal((self.batch_per_device // 2,) + self.initial_img.shape)
                noise = tf.concat([noise_pos, -noise_pos], axis=0)
                eval_points = self.x + self.cfg.sigma * noise
                losses, noise = loss_fn(eval_points, noise)
            losses_tiled = tf.tile(tf.reshape(losses, (-1, 1, 1, 1)), (1,) + self.initial_img.shape)
            grad_estimates.append(tf.reduce_mean(losses_tiled * noise, axis=0) / self.cfg.sigma)
            final_losses.append(losses)

        self.grad_estimate = tf.reduce_mean(grad_estimates, axis=0)
        self.final_losses = tf.concat(final_losses, axis=0)

        # TensorBoard
        self.empirical_loss = tf.placeholder(dtype=tf.float32, shape=())
        self.lr_placeholder = tf.placeholder(dtype=tf.float32, shape=())
        self.loss_vs_queries = tf.summary.scalar("empirical loss vs queries", self.empirical_loss)
        self.loss_vs_steps = tf.summary.scalar("empirical loss vs step", self.empirical_loss)
        self.lr_vs_queries = tf.summary.scalar("lr vs queries", self.lr_placeholder)
        self.lr_vs_steps = tf.summary.scalar("lr vs step", self.lr_placeholder)

        self.writer = tf.summary.FileWriter(self.cfg.out_dir, graph=self.sess.graph)
        self.log_file = open(os.path.join(self.cfg.out_dir, "log.txt"), "w+")

        if self.cfg.visualize:
            with tf.device("/cpu:0"):
                self.render_feed = tf.placeholder(tf.float32, self.initial_img.shape)
                self.render_logits, _ = model(self.sess, tf.expand_dims(self.render_feed, axis=0))

    def _estimate_grad(self, pt):
        """Ước lượng gradient bằng nhiều mini-batch truy vấn."""
        num_batches = self.cfg.samples_per_draw // self.cfg.batch_size
        losses = []
        grads = []
        feed_dict = {self.x: pt}
        for _ in range(num_batches):
            loss, dl_dx = self.sess.run([self.final_losses, self.grad_estimate], feed_dict)
            losses.append(np.mean(loss))
            grads.append(dl_dx)
        return np.array(losses).mean(), np.mean(np.array(grads), axis=0)

    def _robust_in_top_k(self, proposed_adv):
        """Điều kiện chấp nhận bước ở partial-information."""
        if self.k == NUM_LABELS:
            return True
        eval_logits = self.sess.run(self.eval_logits, {self.x: proposed_adv})[0]
        return self.target_class in eval_logits.argsort()[-self.k :][::-1]

    def run(self):
        """Chạy tối ưu chính và ghi toàn bộ output như attacks.py."""
        with open(os.path.join(self.cfg.out_dir, "args.json"), "w") as args_file:
            json.dump(self.cfg.__dict__, args_file)

        num_queries = 0
        g = 0
        last_ls = []
        max_lr = self.cfg.max_lr
        max_iters = int(np.ceil(self.cfg.max_queries // self.cfg.samples_per_draw))

        for i in range(max_iters):
            start = time.time()

            if self.cfg.visualize:
                render_frame(self.sess, self.adv, i, self.render_logits, self.render_feed, self.cfg.out_dir)

            padv = self.sess.run(self.eval_percent_adv, feed_dict={self.x: self.adv})
            if padv == 1 and self.epsilon <= self.goal_epsilon:
                print("[log] early stopping at iteration %d" % i)
                break

            prev_g = g
            l, g = self._estimate_grad(self.adv)
            g = self.cfg.momentum * prev_g + (1.0 - self.cfg.momentum) * g

            # Plateau annealing
            last_ls.append(l)
            last_ls = last_ls[-self.cfg.plateau_length :]
            if len(last_ls) == self.cfg.plateau_length and last_ls[-1] > last_ls[0] and max_lr > self.cfg.min_lr:
                print("[log] Annealing max_lr")
                max_lr = max(max_lr / self.cfg.plateau_drop, self.cfg.min_lr)
                last_ls = []

            # Line search + epsilon decay
            current_lr = max_lr
            prop_de = self.delta_epsilon if (l < self.cfg.adv_thresh and self.epsilon > self.goal_epsilon) else 0.0

            while current_lr >= self.cfg.min_lr:
                if self.k < NUM_LABELS:
                    proposed_epsilon = max(self.epsilon - prop_de, self.goal_epsilon)
                    self.lower = np.clip(self.initial_img - proposed_epsilon, 0, 1)
                    self.upper = np.clip(self.initial_img + proposed_epsilon, 0, 1)

                proposed_adv = self.adv - self.is_targeted * current_lr * np.sign(g)
                proposed_adv = np.clip(proposed_adv, self.lower, self.upper)

                num_queries += 1
                if self._robust_in_top_k(proposed_adv):
                    if prop_de > 0:
                        self.delta_epsilon = max(prop_de, 0.1)
                        last_ls = []
                    self.adv = proposed_adv
                    self.epsilon = max(self.epsilon - prop_de / self.cfg.conservative, self.goal_epsilon)
                    break
                elif current_lr >= self.cfg.min_lr * 2:
                    current_lr = current_lr / 2.0
                else:
                    prop_de = prop_de / 2.0
                    if prop_de == 0:
                        raise ValueError("Did not converge.")
                    if prop_de < 2e-3:
                        prop_de = 0
                    current_lr = max_lr
                    print("[log] backtracking eps to %3f" % (self.epsilon - prop_de,))

            num_queries += self.cfg.samples_per_draw * (self.zero_iters if self.label_only else 1)

            log_text = "Step %05d: loss %.4f lr %.2E eps %.3f (time %.4f)" % (
                i,
                l,
                current_lr,
                self.epsilon,
                time.time() - start,
            )
            self.log_file.write(log_text + "\n")
            print(log_text)

            if i % self.cfg.log_iters == 0:
                lvq, lvs, lrvq, lrvs = self.sess.run(
                    [self.loss_vs_queries, self.loss_vs_steps, self.lr_vs_queries, self.lr_vs_steps],
                    {self.empirical_loss: l, self.lr_placeholder: current_lr},
                )
                self.writer.add_summary(lvq, num_queries)
                self.writer.add_summary(lrvq, num_queries)
                self.writer.add_summary(lvs, i)
                self.writer.add_summary(lrvs, i)

            if (i + 1) % self.cfg.save_iters == 0 and self.cfg.save_iters > 0:
                np.save(os.path.join(self.cfg.out_dir, "%s.npy" % (i + 1)), self.adv)

        log_output(
            self.sess,
            self.eval_logits,
            self.eval_preds,
            self.x,
            self.adv,
            self.initial_img,
            self.target_class,
            self.cfg.out_dir,
            self.orig_class,
            num_queries,
        )

    def config_dict(self):
        """Xuất cấu hình như style _config() của nes_attack.py."""
        return dict(self.cfg.__dict__)


def get_available_devices(requested_gpus=None):
    """Lấy danh sách thiết bị chạy, fallback CPU nếu không có GPU."""
    local_device_protos = device_lib.list_local_devices()
    gpus = [x.name for x in local_device_protos if x.device_type == "GPU"]

    if requested_gpus:
        if requested_gpus > len(gpus):
            raise RuntimeError("not enough GPUs! (requested %d, found %d)" % (requested_gpus, len(gpus)))
        gpus = gpus[:requested_gpus]

    if not gpus:
        print("No GPUs found. Running on CPU.")
        gpus = ["/cpu:0"]

    return gpus


def parse_args():
    """CLI tương thích với main.py để dễ thay thế khi thử nghiệm."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-draw", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--target-class", type=int)
    parser.add_argument("--orig-class", type=int)
    parser.add_argument("--sigma", type=float, default=1e-3)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--img-path", type=str)
    parser.add_argument("--img-index", type=int)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--log-iters", type=int, default=1)
    parser.add_argument("--restore", type=str)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--max-queries", type=int, default=10000)
    parser.add_argument("--save-iters", type=int, default=50)
    parser.add_argument("--plateau-drop", type=float, default=2.0)
    parser.add_argument("--min-lr-ratio", type=int, default=200)
    parser.add_argument("--plateau-length", type=int, default=5)
    parser.add_argument("--gpus", type=int)
    parser.add_argument("--imagenet-path", type=str)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--max-lr", type=float, default=1e-2)
    parser.add_argument("--min-lr", type=float, default=5e-5)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--adv-thresh", type=float, default=-1.0)
    parser.add_argument("--label-only", action="store_true")
    parser.add_argument("--zero-iters", type=int, default=100)
    parser.add_argument("--label-only-sigma", type=float, default=1e-3)
    parser.add_argument("--starting-eps", type=float, default=1.0)
    parser.add_argument("--starting-delta-eps", type=float, default=0.5)
    parser.add_argument("--min-delta-eps", type=float, default=0.1)
    parser.add_argument("--conservative", type=int, default=2)

    args = parser.parse_args()

    if not (
        (args.img_path is None and args.img_index is not None)
        or (args.img_path is not None and args.img_index is None)
    ):
        raise ValueError("can only set one of img-path, img-index")
    if args.img_path and not (args.orig_class or args.target_class):
        raise ValueError("orig and target class required with image path")
    if args.target_class is None and args.img_index is None:
        raise ValueError("must give target class if not using index")
    if args.samples_per_draw % args.batch_size != 0:
        raise ValueError("samples-per-draw must be divisible by batch-size")

    if args.batch_size % 2 != 0:
        raise ValueError("batch-size should be even because NES uses antithetic pairs")

    return args


def main():
    args = parse_args()

    devices = get_available_devices(args.gpus)
    if args.batch_size % (2 * len(devices)) != 0:
        raise ValueError(
            "batch size must be divisible by 2 * number of devices (batch_size=%d, devices=%d)"
            % (args.batch_size, len(devices))
        )

    if os.path.exists(args.out_dir):
        shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir)

    cfg = NESConfig(**vars(args))
    print(json.dumps(cfg.__dict__))

    attack = NESAttackTF1(cfg, devices)
    attack._setup_image_and_classes()
    attack._setup_attack_state()
    attack._build_graph()
    attack.run()


if __name__ == "__main__":
    main()
