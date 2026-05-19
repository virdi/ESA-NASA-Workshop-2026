#!/usr/bin/env python3
"""
Convert PyTorch Lightning checkpoint (.ckpt) to PyTorch binary format (.bin)
that can be loaded by vLLM's terratorch model.
"""

import torch
import argparse
from pathlib import Path
from collections import OrderedDict


def convert_ckpt_to_bin(ckpt_path: str, output_path: str = None, verbose: bool = False):
    """Convert a PyTorch Lightning checkpoint to a .bin file."""
    ckpt_path = Path(ckpt_path)
    
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
    
    if output_path is None:
        output_path = ckpt_path.with_suffix('.bin')
    else:
        output_path = Path(output_path)
    
    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    if verbose:
        print(f"\nCheckpoint keys: {list(checkpoint.keys())}")
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print(f"Found state_dict with {len(state_dict)} parameters")
    else:
        raise ValueError(f"No 'state_dict' key found. Available: {list(checkpoint.keys())}")
    
    cleaned_state_dict = OrderedDict()
    skipped_keys = []
    for key, value in state_dict.items():
        # Skip training-specific parameters that aren't part of the model
        if key.startswith('criterion'):
            skipped_keys.append(key)
            continue
        
        # Keep the full key structure - vLLM expects 'model.encoder', 'model.decoder', etc.
        # Don't remove the 'model.' prefix
        cleaned_state_dict[key] = value
    
    if skipped_keys:
        print(f"\nSkipped {len(skipped_keys)} training-specific parameters:")
        for key in skipped_keys:
            print(f"  - {key}")
    
    if verbose:
        print(f"\nSample parameter names (first 10):")
        for i, key in enumerate(list(cleaned_state_dict.keys())[:10]):
            tensor = cleaned_state_dict[key]
            print(f"  {key}: shape={tensor.shape}, dtype={tensor.dtype}")
    
    output_dict = {'state_dict': cleaned_state_dict}
    
    print(f"\nSaving converted weights to: {output_path}")
    torch.save(output_dict, output_path)
    
    loaded = torch.load(output_path, map_location='cpu')
    if 'state_dict' in loaded:
        print(f"✓ Verification successful: {len(loaded['state_dict'])} parameters saved")
    else:
        print("✗ Verification failed")
    
    print(f"\n✓ Conversion complete!")
    print(f"  Input:  {ckpt_path}")
    print(f"  Output: {output_path}")
    print(f"  Size:   {output_path.stat().st_size / (1024**2):.2f} MB")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Convert Lightning checkpoint to .bin")
    parser.add_argument("ckpt_path", type=str, help="Path to input .ckpt file")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output .bin file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    convert_ckpt_to_bin(args.ckpt_path, args.output, args.verbose)


if __name__ == "__main__":
    main()
