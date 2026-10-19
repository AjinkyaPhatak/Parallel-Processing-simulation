# Verified references (checked 2026-10-19)

Every entry below was confirmed to exist by resolving its DOI through doi.org **and** reading its Crossref
metadata (api.crossref.org), or by opening the arXiv abstract page / the original PDF. Entries that could
not be verified were dropped.

| # | Reference | How verified |
|---|---|---|
| 1 | S. Williams, A. Waterman, D. Patterson. "Roofline: An Insightful Visual Performance Model for Multicore Architectures." *Communications of the ACM* 52(4):65–76, 2009. doi:10.1145/1498765.1498785 | doi.org → dl.acm.org; Crossref metadata (title, authors, vol/iss/pages) |
| 2 | S. Hong, H. Kim. "An Analytical Model for a GPU Architecture with Memory-level and Thread-level Parallelism Awareness." *Proc. 36th Int. Symp. on Computer Architecture (ISCA)*, pp. 152–163, 2009. doi:10.1145/1555754.1555775 | doi.org; Crossref (also reprinted in SIGARCH CAN 37(3), doi:10.1145/1555815.1555775) |
| 3 | V. Volkov. "Better Performance at Lower Occupancy." GPU Technology Conference (GTC), 22 Sept 2010. https://www.nvidia.com/content/GTC-2010/pdfs/2238_GTC2010.pdf | Original 75-slide PDF downloaded; title, author, affiliation (UC Berkeley) and date read from slide 1 |
| 4 | Z. Jia, M. Maggioni, J. Smith, D. P. Scarpazza. "Dissecting the NVidia Turing T4 GPU via Microbenchmarking." arXiv:1903.07486, 2019. | arXiv abstract page |
| 5 | C. Yang, T. Kurth, S. Williams. "Hierarchical Roofline Analysis for GPUs: Accelerating Performance Optimization for the NERSC-9 Perlmutter System." *Concurrency and Computation: Practice and Experience* 32(20):e5547, 2020. doi:10.1002/cpe.5547 | Crossref metadata |
| 6 | P. Hijma, S. Heldens, A. Sclocco, B. van Werkhoven, H. E. Bal. "Optimization Techniques for GPU Programming." *ACM Computing Surveys* 55(11):239:1–239:81, 2023. doi:10.1145/3570638 | Crossref metadata |
| 7 | R. Schoonhoven, B. van Werkhoven, K. J. Batenburg. "Benchmarking Optimization Algorithms for Auto-Tuning GPU Kernels." *IEEE Trans. Evolutionary Computation* 27(3):550–564, 2023. doi:10.1109/TEVC.2022.3210654 | Crossref metadata (preprint arXiv:2210.01465) |
| 8 | R. Schoonhoven, B. Veenboer, B. van Werkhoven, K. J. Batenburg. "Going Green: Optimizing GPUs for Energy Efficiency through Model-Steered Auto-Tuning." *IEEE/ACM PMBS Workshop @ SC22*, pp. 48–59, 2022. doi:10.1109/PMBS56514.2022.00010 | Crossref metadata + arXiv:2211.07260 abstract page |
| 9 | J. O. Tørring, B. van Werkhoven, F. Petrovič, F.-J. Willemsen, J. Filipovič, A. C. Elster. "Towards a Benchmarking Suite for Kernel Tuners." *IEEE IPDPS Workshops (IPDPSW)*, pp. 724–733, 2023. doi:10.1109/IPDPSW59300.2023.00124 | Crossref metadata |

Documentation (not a scholarly paper; used only for the FP32 lanes/SM constant):
NVIDIA, *CUDA C++ Programming Guide*, "Arithmetic Instructions" throughput table, https://docs.nvidia.com/cuda/cuda-c-programming-guide/

Candidate NOT used: Ryoo et al., PPoPP 2008 (not needed; not verified in this session).
