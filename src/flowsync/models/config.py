
eth_prefixes = {'eth_dhost_':48,
                'eth_shost_':48,
                'eth_ethertype_':16}

ipv4_prefixes = {'ipv4_ver_':4,
                    'ipv4_hl_':4,
                    'ipv4_tos_':8,
                    'ipv4_tl_':16,
                    'ipv4_id_':16,
                    'ipv4_rbit_':1, 'ipv4_dfbit_':1, 'ipv4_mfbit_':1,
                    'ipv4_foff_':13,
                    'ipv4_ttl_':8,
                    'ipv4_proto_':8,
                    'ipv4_cksum_':16,
                    'ipv4_src_':32,
                    'ipv4_dst_':32,
                    'ipv4_opt_':320}

ipv6_prefixes = {'ipv6_ver_':4,
                    'ipv6_tc_':8,
                    'ipv6_fl_':20,
                    'ipv6_len_':16,
                    'ipv6_nh_':8,
                    'ipv6_hl_':8,
                    'ipv6_src_':128,
                    'ipv6_dst_':128}

tcp_prefixes = {'tcp_sprt_':16,
                'tcp_dprt_':16,
                'tcp_seq_':32,
                'tcp_ackn_':32,
                'tcp_doff_':4,
                'tcp_res_':3, 'tcp_ns_':1, 'tcp_cwr_':1,
                'tcp_ece_':1, 'tcp_urg_':1,
                'tcp_ackf_':1, 'tcp_psh_':1,
                'tcp_rst_':1, 'tcp_syn_':1,
                'tcp_fin_':1,
                'tcp_wsize_':16,
                'tcp_cksum_':16,
                'tcp_urp_':16,
                'tcp_opt_':320}

udp_prefixes = {'udp_sport_':16,
                'udp_dport_':16,
                'udp_len_':16,
                'udp_cksum_':16}

icmp_prefixes = {'icmp_type_':8,
                    'icmp_code_':8,
                    'icmp_cksum_':8,
                    'icmp_roh_':32}

payload = {} # if "payload_<#>" columns exist, this dictionary will be dynamically populated.
