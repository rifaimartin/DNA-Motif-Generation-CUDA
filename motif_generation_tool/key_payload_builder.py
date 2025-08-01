from payload.payload_builder import PayloadBuilder
from key.key_builder import KeyBuilder

import numpy as np
import time

class KeyPayloadBuilder:
    def __init__(self, constraints, hyperparameters):
        self.constraints = constraints
        self.hyperparameters = hyperparameters

    async def build_keys_and_payloads(self, with_constraints):
        """This function attempts to build keys and payloads respecting the thresholds related to
        the constraints listed in `with_constraints` a total of `key_num` and 
        `payload_num` number of times respectively.
        
        Parameters
        ----------
        with_constraints: set of str
            Set of strings containing a selection of the following constraints: 
            'hairpin', 'hom', 'gcContent'. Those will be the constraints that the
            palyoads will have to conform to.
        
        Returns
        ----------
        keys: list of str or bool
            List of keys conforming to the constraints `with_constraints`. If no such
            key could be generated, it is False.
        payload: set of str or bool
            Set of payloads conforming to the constraints `with_constraints`. If no such
            key could be generated, it is False.
        """

        key_builder = KeyBuilder(self.constraints, self.hyperparameters)
        keys = await key_builder.build_all_keys(with_constraints)
        if not keys:
            return False, False
        payload_builder = PayloadBuilder(self.constraints, self.hyperparameters)
        await payload_builder.add_keys(keys)
        payloads = await payload_builder.build_all_payloads(with_constraints)
        if not payloads:
            return False, False
        return keys, payloads

    def get_motifs(self, keys, payloads):
        """This function returns the set of motifs corresponding to the keys and payloads inputted.
        
        Parameters
        ----------
        keys: list of str
            List of keys.
        payload: set of str
            Set of payloads.
        
        Returns
        ----------
        motifs: set of str or bool
            Set of all motifs built using the provided keys and payloads.
        """
        motifs = set()
        for payload in payloads:
            for i in range(len(keys)):
                motif1 = keys[i] + payload + keys[i]
                motif2 = keys[i] + payload + keys[(i + 1) % len(keys)]
                motifs.add(motif1)
                motifs.add(motif2)
        return motifs
    

async def main():
    from constraints.constraints import Constraints
    from hyperparameters.hyperparameters import Hyperparameters
    payload_size = 60
    payload_num = 15
    max_hom = 5
    max_hairpin = 1
    loop_size_min = 6
    loop_size_max = 7
    min_gc = 25
    max_gc = 65
    key_size = 20
    key_num = 8
    
    constraints = Constraints(payload_size=payload_size, payload_num=payload_num, \
                              max_hom=max_hom, max_hairpin=max_hairpin, \
                              min_gc=min_gc, max_gc=max_gc, key_size=key_size, \
                              key_num=key_num, loop_size_min=loop_size_min, \
                              loop_size_max=loop_size_max)

    num_rounds = 10000
    num_successful_motifs = 0
    with_constraints = {'hom', 'gcContent', 'hairpin', 'noKeyInPayload'}
    
    # Store successful results
    all_successful_keys = []
    all_successful_payloads = []
    all_successful_motifs = []
    
    # Track start time
    start_time = time.time()
    
    print(f"Starting motif generation with {num_rounds} rounds...")
    print(f"Constraints: {with_constraints}")
    print("-" * 60)
    
    for i in range(num_rounds):
        # Progress indicator - show every 100 iterations
        if i % 100 == 0 and i > 0:
            elapsed_time = time.time() - start_time
            success_rate = (num_successful_motifs / i) * 100
            avg_time_per_round = elapsed_time / i
            estimated_total_time = avg_time_per_round * num_rounds
            remaining_time = estimated_total_time - elapsed_time
            
            print(f"Progress: {i:5d}/{num_rounds} ({i/num_rounds*100:5.1f}%) | "
                  f"Success: {num_successful_motifs:4d} ({success_rate:5.1f}%) | "
                  f"Elapsed: {elapsed_time:6.1f}s | "
                  f"ETA: {remaining_time:6.1f}s")
        
        for weight in [1]:
            shapes = {'hom': 70, 'gcContent': 10, 'hairpin': 8, 'similarity': 60, 'noKeyInPayload': 45}
            weights = {'hom': 1, 'gcContent': 1, 'hairpin': 1, 'similarity': 1, 'noKeyInPayload': 1}
            hyperparams = Hyperparameters(shapes, weights)

            keyPayloadBuilder = KeyPayloadBuilder(constraints, hyperparams)
            keys, payloads = await keyPayloadBuilder.build_keys_and_payloads(with_constraints)
            if not (keys and payloads):
              continue
              
            # Store successful results
            motifs = keyPayloadBuilder.get_motifs(keys, payloads)
            all_successful_keys.append(keys)
            all_successful_payloads.append(payloads)
            all_successful_motifs.append(motifs)
            
            num_successful_motifs += 1

    # Final results
    total_time = time.time() - start_time
    success_rate = (num_successful_motifs / num_rounds) * 100
    
    print("-" * 60)
    print(f"COMPLETED!")
    print(f"Total rounds: {num_rounds}")
    print(f"Successful motif sets: {num_successful_motifs}")
    print(f"Success rate: {success_rate:.2f}%")
    print(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"Average time per round: {total_time/num_rounds:.3f} seconds")
    
    # Save results to files
    if num_successful_motifs > 0:
        print("\nSaving results to files...")
        
        # Save first few successful examples
        with open("successful_keys.txt", "w") as f:
            for i, keys in enumerate(all_successful_keys[:10]):  # Save first 10
                f.write(f"Keys Set {i+1}: {keys}\n")
        
        with open("successful_payloads.txt", "w") as f:
            for i, payloads in enumerate(all_successful_payloads[:10]):
                f.write(f"Payloads Set {i+1}: {list(payloads)}\n")
        
        with open("successful_motifs.txt", "w") as f:
            for i, motifs in enumerate(all_successful_motifs[:10]):
                f.write(f"Motifs Set {i+1}:\n")
                for motif in list(motifs)[:20]:  # Save first 20 motifs per set
                    f.write(f"  {motif}\n")
                f.write("\n")
        
        print("Results saved to:")
        print("- successful_keys.txt")
        print("- successful_payloads.txt") 
        print("- successful_motifs.txt")
        print(f"(Showing first 10 successful sets)")
        
        # Show sample results
        print(f"\nSample from first successful set:")
        print(f"Keys: {all_successful_keys[0]}")
        print(f"Payloads: {list(all_successful_payloads[0])[:3]}...")  # Show first 3 payloads
        print(f"Motifs: {list(all_successful_motifs[0])[:5]}...")      # Show first 5 motifs

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())