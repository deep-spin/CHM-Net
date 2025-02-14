import torchvision
import random
from PIL import Image, ImageOps
import numpy as np
import numbers
import math
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from decord import VideoReader, cpu
from torchvision import datasets, transforms
import torch.distributions as dist

class GroupRandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img_group):

        w, h = img_group[0].size
        th, tw = self.size

        out_images = list()

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        for img in img_group:
            assert(img.size[0] == w and img.size[1] == h)
            if w == tw and h == th:
                out_images.append(img)
            else:
                out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))

        return out_images


class MultiGroupRandomCrop(object):
    def __init__(self, size, groups=1):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.groups = groups

    def __call__(self, img_group):

        w, h = img_group[0].size
        th, tw = self.size

        out_images = list()

        for i in range(self.groups):
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            for img in img_group:
                assert(img.size[0] == w and img.size[1] == h)
                if w == tw and h == th:
                    out_images.append(img)
                else:
                    out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))

        return out_images


class GroupCenterCrop(object):
    def __init__(self, size):
        self.worker = torchvision.transforms.CenterCrop(size)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupRandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __init__(self, is_flow=False):
        self.is_flow = is_flow

    def __call__(self, img_group, is_flow=False):
        v = random.random()
        if v < 0.5:
            ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            if self.is_flow:
                for i in range(0, len(ret), 2):
                    # invert flow pixel values when flipping
                    ret[i] = ImageOps.invert(ret[i])
            return ret
        else:
            return img_group


class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        rep_mean = self.mean * (tensor.size()[0] // len(self.mean))
        rep_std = self.std * (tensor.size()[0] // len(self.std))

        # TODO: make efficient
        for t, m, s in zip(tensor, rep_mean, rep_std):
            t.sub_(m).div_(s)

        return tensor


class GroupScale(object):
    """ Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.worker = torchvision.transforms.Resize(size, interpolation)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupOverSample(object):
    def __init__(self, crop_size, scale_size=None, flip=True):
        self.crop_size = crop_size if not isinstance(
            crop_size, int) else (crop_size, crop_size)

        if scale_size is not None:
            self.scale_worker = GroupScale(scale_size)
        else:
            self.scale_worker = None
        self.flip = flip

    def __call__(self, img_group):

        if self.scale_worker is not None:
            img_group = self.scale_worker(img_group)

        image_w, image_h = img_group[0].size
        crop_w, crop_h = self.crop_size

        offsets = GroupMultiScaleCrop.fill_fix_offset(
            False, image_w, image_h, crop_w, crop_h)
        oversample_group = list()
        for o_w, o_h in offsets:
            normal_group = list()
            flip_group = list()
            for i, img in enumerate(img_group):
                crop = img.crop((o_w, o_h, o_w + crop_w, o_h + crop_h))
                normal_group.append(crop)
                flip_crop = crop.copy().transpose(Image.FLIP_LEFT_RIGHT)

                if img.mode == 'L' and i % 2 == 0:
                    flip_group.append(ImageOps.invert(flip_crop))
                else:
                    flip_group.append(flip_crop)

            oversample_group.extend(normal_group)
            if self.flip:
                oversample_group.extend(flip_group)
        return oversample_group


class GroupFullResSample(object):
    def __init__(self, crop_size, scale_size=None, flip=True):
        self.crop_size = crop_size if not isinstance(
            crop_size, int) else (crop_size, crop_size)

        if scale_size is not None:
            self.scale_worker = GroupScale(scale_size)
        else:
            self.scale_worker = None
        self.flip = flip

    def __call__(self, img_group):

        if self.scale_worker is not None:
            img_group = self.scale_worker(img_group)

        image_w, image_h = img_group[0].size
        crop_w, crop_h = self.crop_size

        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        offsets = list()
        offsets.append((0 * w_step, 2 * h_step))  # left
        offsets.append((4 * w_step, 2 * h_step))  # right
        offsets.append((2 * w_step, 2 * h_step))  # center

        oversample_group = list()
        for o_w, o_h in offsets:
            normal_group = list()
            flip_group = list()
            for i, img in enumerate(img_group):
                crop = img.crop((o_w, o_h, o_w + crop_w, o_h + crop_h))
                normal_group.append(crop)
                if self.flip:
                    flip_crop = crop.copy().transpose(Image.FLIP_LEFT_RIGHT)

                    if img.mode == 'L' and i % 2 == 0:
                        flip_group.append(ImageOps.invert(flip_crop))
                    else:
                        flip_group.append(flip_crop)

            oversample_group.extend(normal_group)
            oversample_group.extend(flip_group)
        return oversample_group


class GroupMultiScaleCrop(object):

    def __init__(self, input_size, scales=None, max_distort=1,
                 fix_crop=True, more_fix_crop=True):
        self.scales = scales if scales is not None else [1, .875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [
            input_size, input_size]
        self.interpolation = Image.BILINEAR

    def __call__(self, img_group):

        im_size = img_group[0].size

        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = [
            img.crop(
                (offset_w,
                 offset_h,
                 offset_w +
                 crop_w,
                 offset_h +
                 crop_h)) for img in img_group]
        ret_img_group = [img.resize((self.input_size[0], self.input_size[1]), self.interpolation)
                         for img in crop_img_group]
        return ret_img_group

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]

        # find a crop size
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [
            self.input_size[1] if abs(
                x - self.input_size[1]) < 3 else x for x in crop_sizes]
        crop_w = [
            self.input_size[0] if abs(
                x - self.input_size[0]) < 3 else x for x in crop_sizes]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(
                image_w, image_h, crop_pair[0], crop_pair[1])

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(
            self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        if more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

        return ret


class GroupRandomSizedCrop(object):
    """Random crop the given PIL.Image to a random size of (0.08 to 1.0) of the original size
    and and a random aspect ratio of 3/4 to 4/3 of the original aspect ratio
    This is popularly used to train the Inception networks
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img_group):
        for attempt in range(10):
            area = img_group[0].size[0] * img_group[0].size[1]
            target_area = random.uniform(0.08, 1.0) * area
            aspect_ratio = random.uniform(3. / 4, 4. / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img_group[0].size[0] and h <= img_group[0].size[1]:
                x1 = random.randint(0, img_group[0].size[0] - w)
                y1 = random.randint(0, img_group[0].size[1] - h)
                found = True
                break
        else:
            found = False
            x1 = 0
            y1 = 0

        if found:
            out_group = list()
            for img in img_group:
                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert(img.size == (w, h))
                out_group.append(
                    img.resize(
                        (self.size, self.size), self.interpolation))
            return out_group
        else:
            # Fallback
            scale = GroupScale(self.size, interpolation=self.interpolation)
            crop = GroupRandomCrop(self.size)
            return crop(scale(img_group))


class ConvertDataFormat(object):
    def __init__(self, model_type):
        self.model_type = model_type

    def __call__(self, images):
        if self.model_type == '2D':
            return images
        tc, h, w = images.size()
        t = tc // 3
        images = images.view(t, 3, h, w)
        images = images.permute(1, 0, 2, 3)
        return images


class Stack(object):

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if img_group[0].mode == 'L':
            return np.concatenate([np.expand_dims(x, 2)
                                   for x in img_group], axis=2)
        elif img_group[0].mode == 'RGB':
            if self.roll:
                return np.concatenate([np.array(x)[:, :, ::-1]
                                       for x in img_group], axis=2)
            else:
                #print(np.concatenate(img_group, axis=2).shape)
                # print(img_group[0].shape)
                return np.concatenate(img_group, axis=2)


class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """

    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(
                    pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255) if self.div else img.float()


class IdentityTransform(object):

    def __call__(self, data):
        return data


def get_index(num_frames, num_segments):
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array([
        start + int(np.round(seg_size * idx)) for idx in range(num_segments)
    ])
    return offsets


def load_video(video_path, num_segments=8, resolution=224):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    num_frames = len(vr)
    frame_indices = get_index(num_frames, num_segments)

    # transform
    crop_size = resolution
    scale_size = resolution

    transform = T.Compose([
        GroupScale(int(scale_size), interpolation=InterpolationMode.BICUBIC),
        GroupCenterCrop(crop_size),
        Stack(),
        ToTorchFormatTensor(),
        transforms.Normalize(0.5,0.5,0.5)
    ])

    images_group = list()
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy())
        images_group.append(img)
    torch_imgs = transform(images_group)
    return torch_imgs
    
class BasisFunctions(object):
    def __init__(self):
        pass

    def __len__(self):
        """Number of basis functions."""
        pass

    def evaluate(self, t):
        pass

    def integrate_t2_times_psi(self, a, b):
        """Compute integral int_a^b (t**2) * psi(t)."""
        pass

    def integrate_t_times_psi(self, a, b):
        """Compute integral int_a^b t * psi(t)."""
        pass

    def integrate_psi(self, a, b):
        """Compute integral int_a^b psi(t)."""
        pass
class RetangularBasisFunctions(BasisFunctions):
    """Function phi(t) = Gaussian(t; mu, sigma_sq)."""
    def __init__(self, mu, sigma):
        self.mu = mu.unsqueeze(0)
        self.width = sigma.unsqueeze(0)

    def __repr__(self):
        return f"GaussianBasisFunction(mu={self.mu}, sigma={self.sigma})"

    def __len__(self):
        """Number of basis functions."""
        return self.mu.size(1)
    
    def full_batch_evaluate(self, t):
        """
        Evaluate multiple time points against all rectangular basis functions.
        Args:
            t: Tensor of time values to evaluate, shape (batch_size, num_points).
        Returns:
            Tensor of evaluations, shape (batch_size, num_basis, num_points).
        """
        # Repeat mu and width across the batch dimension
        mu = self.mu.repeat(t.size(0), 1)  # Shape: (batch_size, num_basis)
        width = self.width.repeat(t.size(0), 1)  # Shape: (batch_size, num_basis)
        
        # Now evaluate the condition for each time point in the batch
        result = ((t.unsqueeze(2) >= (mu - width / 2).unsqueeze(1)) &
                (t.unsqueeze(2) < (mu + width / 2).unsqueeze(1))).float()
        
        return result
    
    def batch_evaluate(self, t):
        """
        Evaluate multiple time points against all rectangular basis functions.
        Args:
            t: Tensor of time values to evaluate, shape (num_points,).
        Returns:
            Tensor of evaluations, shape (num_basis, num_points).
        """
        t = t.repeat(self.mu.size(0),1)  # Shape: (num_basis, num_points)
        mu = self.mu.repeat(t.size(0),1).transpose(1,0)  # Shape: (num_basis, num_points)
        width = self.width.repeat(t.size(0),1).transpose(1,0)  # Shape: (num_basis,num_points)
        return ((t >= (mu - width / 2)) & (t < (mu + width / 2))).float().transpose(0,1)
    
    def _Phi(self, t):
        """
        Compute the step function for a single value of t.
        Args:
            t: A scalar or tensor of time values.
        Returns:
            Tensor of values indicating presence in each basis function's range.
        """
        lower_bounds = self.mu - self.width / 2
        upper_bounds = self.mu + self.width / 2
        return ((t >= lower_bounds) & (t < upper_bounds)).float()

    def evaluate(self, t):
        """
        Evaluate the rectangular basis functions at a single point or array of points.
        Args:
            t: A scalar or 1D tensor of time values.
        Returns:
            Tensor of shape (num_basis,) for scalar input, or (num_basis, num_points) for tensor input.
        """
        if t.ndim == 0:  # Scalar input
            return self._Phi(t)
        else:  # Tensor input
          # Shape: (1, num_points)
            lower_bounds = (self.mu - self.width / 2)  # Shape: (num_basis, 1)
            upper_bounds = (self.mu + self.width / 2)  # Shape: (num_basis, 1)
            return ((t >= lower_bounds) & (t < upper_bounds)).float()
        
class ContinuousHopfieldNet(torch.nn.Module):
    
    def __init__(self, beta, nb_basis, num_iters, num_points, device):
        super(ContinuousHopfieldNet, self).__init__()
        self.beta =beta
        self.nb_basis = nb_basis
        self.device = device
        self.num_points = num_points
        self.num_iters = num_iters
        self.ridge_penalty=0.5 # ridge penalty
        self.spacing='linear'
    
    def get_basis(self, length, target_len):
        def compute_G(l, psi, positions, padding=True):

            F = torch.zeros(self.nb_basis, positions.size(0))

            basis_functions = psi
            F[:, :] = basis_functions.evaluate(positions.unsqueeze(1)).t()
            I = torch.eye(self.nb_basis)
            G = F.t().matmul((F.matmul(F.t()) + self.ridge_penalty * I).inverse())

            if padding:
                if l % 2:
                    G = G[((l-1)//2):(-(l-1)//2), :]
                else:
                    G = G[(l//2):-(l//2), :]

            return G.to(self.device)
        padding = True
        self.psi=[None]
        self.Gs=[None for _ in range(length+1)]
        lengths=[]
        for i in range(length):
            self.psi.append([])
            if (i+1)%target_len==0:
                lengths.append(i+1)
        if length not in lengths:
            lengths.append(length)
        for l in lengths:
            # get positions for memory vectors
            self.add_retangular_basis_functions(self.psi[l])

            if self.spacing=='linear':
                if padding:
                    if l % 2:
                        shift = 1 / float(l)
                        positions = torch.linspace(-.5+shift, 1.5-shift, 2*l-1).to(self.device)
                    else:
                        shift = 1 / float(2*l)
                        positions = torch.linspace(-.5+shift, 1.5-shift, 2*l).to(self.device)
                else:
                    shift = 1 / float(2*l)
                    positions = torch.linspace(shift, 1-shift, l).to(self.device)
            elif self.spacing=='log':
                if padding:
                    if l % 2:
                        shift = 1 / float(l)
                        positions = torch.linspace(-.5+shift, 1.5-shift, 2*l-1).to(self.device)
                    else:
                        shift = 1 / float(2*l)
                        positions = torch.linspace(-.5+shift, 1.5-shift, 2*l).to(self.device)

                    pos = np.e**(np.log(1+1)*torch.arange(1,length+1)/length)-1
                    positions = torch.cat([positions[:int(l/2)],pos.to(self.device),positions[-int(l/2):]])

                else:
                    positions = np.e**(np.log(1+1)*torch.arange(1,length+1)/length)-1
        
            self.Gs[l]=compute_G(l, self.psi[l][0], positions, padding=padding) # [L,N]
    
    def add_retangular_basis_functions(self, psi):
        width = torch.ones(self.nb_basis, device=self.device) / self.nb_basis
    
        # Compute the centers (midpoints) of each bin
        edges = torch.linspace(0, 1, self.nb_basis + 1, device=self.device)
        mu = (edges[:-1] + edges[1:]) / 2 
        psi.append(RetangularBasisFunctions(mu=mu, sigma=width))
    
    def value_function(self, x, inf=False):
        G = self.Gs[x.size(-1)] # [L,N]
        B = torch.matmul(x, G) # [B,e,N]
        B = B.permute(0,2,1) # [B,N,e]
        
        return B
    
    def score(self, t):
        psis = self.psis[0].batch_evaluate(t)
        query = self.beta*self.queries # divide by sqrt(d_head) [B,h,q,d]
        keys = self.keys.transpose(-1, -2)
        keys = keys.mm(psis.T) #[B,h,d,1]
        scores = query.mm(keys) #[B,h,q,1] 
        return scores

    def compute_probability(self, score_fn, t=None):
        """
        Compute probability distribution p(t).
        
        Args:
            score_fn (callable): Function that computes z(t)
            num_points (int): Number of points for numerical integration
        
        Returns:
            tuple: (probabilities, normalization constant)
        """
        num_points = self.num_points
        if t is None:
            # Create integration points
            t = torch.linspace(0, 1, num_points).to(self.device)
        scores = score_fn(t)
        maxes = torch.max(scores, 1, keepdim=True)[0]
        prob = torch.exp(scores-maxes) / torch.trapz(torch.exp(scores-maxes), t, dim=-1).unsqueeze(-1)
        return prob
    
    def expected_value(self, score_fn):
        """
        Compute expected value E_p[V(t)] using nested integration.
        
        Args:
            score_fn (callable): Function that computes z(t)
            value_fn (callable): Function that computes v(t)
            num_points (int): Number of points for numerical integration
        
        Returns:
            torch.Tensor: Expected value
        """
        num_points = self.num_points
        # Create integration points
        t = torch.linspace(0, 1, num_points).to(self.device)
        
        # Compute basis functions
        self.psis = []
        self.add_retangular_basis_functions(self.psis)
        psi = self.psis[0].batch_evaluate(t)
        prob = self.compute_probability(score_fn)
        values = self.values
        psi_broadcasted = psi.expand(num_points, self.nb_basis)
        integrand = torch.matmul(prob.T.unsqueeze(-1), psi_broadcasted.unsqueeze(1))
        integral  = torch.trapz(integrand, t, dim=0)
        expected_value = integral.mm(values)  # [B, h, q, d]
        
        return expected_value

    def forward(self, k, q, return_contexts=False):
        klen = k.size(0)
        self.length = klen
        batch_size = k.size(0) #batch size
        qlen = q.size(1) #query length
        self.qlen = qlen
        self.batch_size = batch_size
        self.d = k.size(-1)
        self.get_basis(klen, klen)
        # clean memory if going through different document
        k = k.transpose(0,1).unsqueeze(0)
        # perform memory update
        B = self.value_function(k).squeeze(0) # [B,N,e]
        self.keys = B
        self.values = B
        self.queries = q
        contexts = []
        for _ in range(self.num_iters):
            context = self.expected_value(self.score)  # Shape [1, 32, 768]
            contexts.append(context)
            self.queries = context
        if return_contexts:
            return context.contiguous(), torch.stack(contexts).squeeze(1)
        return context.contiguous()
    
class DiscreteHopfieldNet(torch.nn.Module):
    
    def __init__(self, beta, num_iters, device):
        super(DiscreteHopfieldNet, self).__init__()
        self.beta =beta
        self.device = device
        self.num_iters = num_iters
    
    def forward(self, k, q):
        for _ in range(self.num_iters):
            scores = q.mm(k.T) #[B,h,q,1]
            p = torch.softmax(self.beta*scores, dim= -1)
            out = p.mm(k)
        return out

def energy(beta, x, q, model=False):
    if model:
        t = torch.linspace(0, 1, 500).to(x.device)
        model.get_basis(x.size(0), x.size(0))
        k = x.transpose(0,1).unsqueeze(0)
        B = model.value_function(k).squeeze(0) # [B,N,e]
        model.psis = []
        model.add_retangular_basis_functions(model.psis)
        model.keys = B
        model.values = B
        model.queries = q
        score = model.score(t)
        model.score(t)
        E = -torch.log(torch.trapz(torch.exp(score), t, dim=-1)/beta) + 0.5*(q**2).sum(dim=-1)
    else:
        E = -torch.logsumexp(beta*q@x.T, 1, keepdim=False)/beta + 0.5*(q**2).sum(dim=-1)
    return E
