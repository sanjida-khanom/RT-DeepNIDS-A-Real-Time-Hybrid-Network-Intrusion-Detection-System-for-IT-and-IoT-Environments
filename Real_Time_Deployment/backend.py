# =============================================================================
# RT-DeepNIDS — backend.py
# A Real-Time Hybrid Network Intrusion Detection System
# Bangladesh University of Business and Technology (BUBT)
# Department of Computer Science and Engineering
#
# Research Team:
#   Ayesha Siddika         (22234103099)
#   Sanjida Khanom         (22234103103)
#   Ihsanul Hossain Rafsan (22234103112)
#   Sadia Mehrin Rahi      (22234103122)
#   Istiyak Hasan Maruf    (22234103130)
#
# This module contains all backend logic:
#   - Privilege / admin check
#   - Dataset column templates (CIC-IDS-2017, CIC-IDS-2018, ToN-IoT-v3)
#   - FlowTracker: raw Scapy packets -> dataset-aligned feature vectors
#   - PortScanDetector: cross-flow heuristic for nmap-style SYN scans
#   - Model / scaler / CSV loaders
#   - Label cleaning utilities
# =============================================================================


import os
import sys
import re
import time as _time


import numpy as np
import pandas as pd
import joblib

# Scapy — optional; graceful fallback when not installed

try:
   from scapy.all import sniff, IP, TCP, UDP, get_if_list
   SCAPY_AVAILABLE = True
except ImportError:
   SCAPY_AVAILABLE = False
   get_if_list = lambda: []

# Admin / privilege check
def is_admin() -> bool:
   """Return True if the process has Administrator (Windows) or root (Linux/macOS) privileges."""
   try:
       if sys.platform == "win32":
           import ctypes
           return bool(ctypes.windll.shell32.IsUserAnAdmin())
       else:
           return os.getuid() == 0
   except Exception:
       return False


IS_ADMIN: bool = is_admin()


# Directory & model mappings
BASE_DIR = os.path.join(os.path.dirname(__file__), "Real_Time_Export")


DATASET_FOLDER_MAP = {
   "CIC-IDS-2017": "CIC_IDS_2017",
   "CIC-IDS-2018": "CIC_IDS_2018",
   "ToN-IoT-v3":   "TON_IOT_V3",
}
BALANCING_FOLDER_MAP = {
   "SMOTE":     "SMOTE",
   "Tomek+IHT": "Tomek_IHT",
   "Tomek_IHT": "Tomek_IHT",   # accept both spellings defensively
}
MODEL_FILE_MAP = {
   "Hybrid CNN-GRU":  "CNN_GRU_model.h5",
   "Decision Tree":   "DT_model.pkl",
   "Random Forest":   "RF_model.pkl",
   "XGBoost":         "XGB_model.pkl",
   "CNN+Transformer": "CNN_Transformer_model.h5",
}
DL_MODELS = {"Hybrid CNN-GRU", "CNN+Transformer"}


EXCLUDE_COLS = {
   "Label", "Class", "Attack", "type", "category",
   "Timestamp", "Source IP", "Dest IP",
   "source_ip", "dest_ip", "src_ip", "dst_ip",
}

# Dataset column templates
# DESIGN NOTE:
#   Feature POSITION matters — CICFlowMeter exports have a strict column order.
#   Each template maps column_name -> vector_index; values are placed at those
#   exact indices so the trained scaler / model sees the correct layout.


_CIC2017_COLS = [
   " Destination Port", " Flow Duration", " Total Fwd Packets",
   " Total Backward Packets", "Total Length of Fwd Packets",
   " Total Length of Bwd Packets", " Fwd Packet Length Max",
   " Fwd Packet Length Min", " Fwd Packet Length Mean",
   " Fwd Packet Length Std", "Bwd Packet Length Max",
   " Bwd Packet Length Min", " Bwd Packet Length Mean",
   " Bwd Packet Length Std", "Flow Bytes/s", " Flow Packets/s",
   " Flow IAT Mean", " Flow IAT Std", " Flow IAT Max", " Flow IAT Min",
   "Fwd IAT Total", " Fwd IAT Mean", " Fwd IAT Std", " Fwd IAT Max",
   " Fwd IAT Min", "Bwd IAT Total", " Bwd IAT Mean", " Bwd IAT Std",
   " Bwd IAT Max", " Bwd IAT Min", "Fwd PSH Flags", " Bwd PSH Flags",
   " Fwd URG Flags", " Bwd URG Flags", " Fwd Header Length",
   " Bwd Header Length", "Fwd Packets/s", " Bwd Packets/s",
   " Min Packet Length", " Max Packet Length", " Packet Length Mean",
   " Packet Length Std", " Packet Length Variance", "FIN Flag Count",
   " SYN Flag Count", " RST Flag Count", " PSH Flag Count",
   " ACK Flag Count", " URG Flag Count", " CWE Flag Count",
   " ECE Flag Count", " Down/Up Ratio", " Average Packet Size",
   " Avg Fwd Segment Size", " Avg Bwd Segment Size",
   " Fwd Header Length.1", "Fwd Avg Bytes/Bulk", " Fwd Avg Packets/Bulk",
   " Fwd Avg Bulk Rate", " Bwd Avg Bytes/Bulk", " Bwd Avg Packets/Bulk",
   "Bwd Avg Bulk Rate", "Subflow Fwd Packets", " Subflow Fwd Bytes",
   " Subflow Bwd Packets", " Subflow Bwd Bytes", "Init_Win_bytes_forward",
   " Init_Win_bytes_backward", " act_data_pkt_fwd",
   " min_seg_size_forward", "Active Mean", " Active Std",
   " Active Max", " Active Min", "Idle Mean", " Idle Std",
   " Idle Max", " Idle Min",
]  # 78 features


_CIC2018_COLS = [
   "Dst Port", "Protocol", "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts",
   "TotLen Fwd Pkts", "TotLen Bwd Pkts", "Fwd Pkt Len Max", "Fwd Pkt Len Min",
   "Fwd Pkt Len Mean", "Fwd Pkt Len Std", "Bwd Pkt Len Max", "Bwd Pkt Len Min",
   "Bwd Pkt Len Mean", "Bwd Pkt Len Std", "Flow Byts/s", "Flow Pkts/s",
   "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
   "Fwd IAT Tot", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
   "Bwd IAT Tot", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
   "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
   "Fwd Header Len", "Bwd Header Len", "Fwd Pkts/s", "Bwd Pkts/s",
   "Pkt Len Min", "Pkt Len Max", "Pkt Len Mean", "Pkt Len Std", "Pkt Len Var",
   "FIN Flag Cnt", "SYN Flag Cnt", "RST Flag Cnt", "PSH Flag Cnt",
   "ACK Flag Cnt", "URG Flag Cnt", "CWE Flag Cnt", "ECE Flag Cnt",
   "Down/Up Ratio", "Pkt Size Avg", "Fwd Seg Size Avg", "Bwd Seg Size Avg",
   "Fwd Byts/b Avg", "Fwd Pkts/b Avg", "Fwd Blk Rate Avg",
   "Bwd Byts/b Avg", "Bwd Pkts/b Avg", "Bwd Blk Rate Avg",
   "Subflow Fwd Pkts", "Subflow Fwd Byts", "Subflow Bwd Pkts", "Subflow Bwd Byts",
   "Init Fwd Win Byts", "Init Bwd Win Byts", "Fwd Act Data Pkts",
   "Fwd Seg Size Min", "Active Mean", "Active Std", "Active Max", "Active Min",
   "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]  # 80 features


# ToN-IoT-v3 core Zeek/Bro flow fields (47).
# The full ~470-column space is built by _expand_ton_cols() below,
# which appends one-hot / label-encoded expansion columns to match
# the training pipeline's pd.get_dummies() output.
_TON_COLS_CORE = [
   "duration", "proto", "src_port", "dst_port", "src_pkts", "src_bytes",
   "dst_pkts", "dst_bytes", "conn_state", "missed_bytes", "src_ip_bytes",
   "dst_ip_bytes", "dns_query", "dns_qclass", "dns_qtype", "dns_rcode",
   "dns_AA", "dns_RD", "dns_RA", "dns_rejected",
   "ssl_version", "ssl_cipher", "ssl_resumed", "ssl_established",
   "ssl_subject", "ssl_issuer",
   "http_trans_depth", "http_method", "http_uri", "http_referrer",
   "http_version", "http_request_body_len", "http_response_body_len",
   "http_status_code", "http_user_agent", "http_orig_mime_types",
   "http_resp_mime_types",
   "weird_name", "weird_addl", "weird_notice",
   "service", "duration_log1p", "src_bytes_log1p", "dst_bytes_log1p",
   "src_pkts_log1p", "dst_pkts_log1p", "missed_bytes_log1p",
]  # 47 core features


_TON_PROTO_VALS = [
   "tcp", "udp", "icmp", "arp", "ospf", "rarp", "ipv6-icmp", "ip",
   "igmp", "gre", "esp", "ah", "ipv6", "sctp", "other",
]
_TON_CONNSTATE_VALS = [
   "S0", "S1", "SF", "REJ", "S2", "S3", "RSTO", "RSTR", "RSTOS0",
   "RSTRH", "SH", "SHR", "OTH",
]
_TON_SERVICE_VALS = [
   "http", "ftp", "ftp-data", "smtp", "dns", "ssh", "ssl", "dhcp",
   "irc", "pop3", "imap", "rdp", "snmp", "smb", "other", "-",
]
_TON_DNSQTYPE_VALS = list(range(0, 66))
_TON_HTTP_METHOD_VALS = [
   "GET", "POST", "HEAD", "PUT", "DELETE", "OPTIONS",
   "PATCH", "CONNECT", "TRACE", "other",
]
_TON_HTTP_STATUS_VALS = [str(c) for c in [
   100, 101, 200, 201, 202, 204, 206, 301, 302, 304, 307, 308,
   400, 401, 403, 404, 405, 408, 409, 410, 413, 415, 422, 429,
   500, 501, 502, 503, 504,
]]
_TON_SSLVER_VALS = ["SSLv3", "TLSv10", "TLSv11", "TLSv12", "TLSv13", "other"]

def _expand_ton_cols() -> list:
   """
   Build the full ~470-column list matching the training pipeline's
   pd.get_dummies() output for ToN-IoT-v3.
   Called once at import time; result cached as _TON_COLS.
   """
   cols = list(_TON_COLS_CORE)
   for v in _TON_PROTO_VALS:       cols.append(f"proto_{v}")
   for v in _TON_CONNSTATE_VALS:   cols.append(f"conn_state_{v}")
   for v in _TON_SERVICE_VALS:     cols.append(f"service_{v}")
   for v in _TON_DNSQTYPE_VALS:    cols.append(f"dns_qtype_{v}")
   for v in _TON_HTTP_METHOD_VALS: cols.append(f"http_method_{v}")
   for v in _TON_HTTP_STATUS_VALS: cols.append(f"http_status_{v}")
   for v in _TON_SSLVER_VALS:      cols.append(f"ssl_version_{v}")
   while len(cols) < 470:
       cols.append(f"ton_feat_{len(cols)}")
   return cols


_TON_COLS = _expand_ton_cols()


DATASET_COLS = {
   "CIC-IDS-2017": _CIC2017_COLS,
   "CIC-IDS-2018": _CIC2018_COLS,
   "ToN-IoT-v3":   _TON_COLS,
   "Cross-Domain": _CIC2017_COLS,  # cross-domain models trained on CIC layout
}


# Conservative benign-traffic imputation values (CICFlowMeter medians).
# Used for columns that cannot be computed from a single raw packet;
# imputing with dataset means keeps scaled values near z=0.
_CIC_IMPUTE = {
   "flow_duration_us": 50_000.0,
   "pkt_len_mean":     512.0,
   "pkt_len_std":      256.0,
   "iat_mean_us":      5_000.0,
   "iat_std_us":       8_000.0,
   "header_len":       20.0,
   "init_win_fwd":     65535.0,
   "init_win_bwd":     65535.0,
   "active_mean":      0.0,
   "idle_mean":        0.0,
}


# Minimum flow duration used as denominator floor (100 ms by default;
# set to 10 ms in High-Sensitivity / Demo Mode via set_min_flow_sec()).
_MIN_FLOW_SEC: float = 0.1

def set_min_flow_sec(val: float) -> None:
   global _MIN_FLOW_SEC
   _MIN_FLOW_SEC = val

# _FlowRecord — per-flow packet accumulator
class _FlowRecord:
   """
   Accumulates per-packet statistics for a single 5-tuple flow and
   produces a dataset-aligned feature vector on demand.
   """


   FLOW_TIMEOUT = 120.0  # seconds; idle flows are evicted by FlowTracker


   __slots__ = (
       "proto", "src_ip", "dst_ip", "sport", "dport",
       "start", "last", "_last_pkt_ts",
       "fwd_pkts", "bwd_pkts", "fwd_bytes", "bwd_bytes",
       "pkt_lengths", "fwd_pkt_lengths", "bwd_pkt_lengths",
       "flow_iat", "fwd_iat", "bwd_iat",
       "flag_fin", "flag_syn", "flag_rst", "flag_psh",
       "flag_ack", "flag_urg", "flag_cwe", "flag_ece",
       "fwd_hdr_bytes", "bwd_hdr_bytes",
       "init_win_fwd", "init_win_bwd",
       "_fwd_win_seen", "_bwd_win_seen",
   )


   def __init__(self, proto, src_ip, dst_ip, sport, dport):
       self.proto  = proto
       self.src_ip = src_ip
       self.dst_ip = dst_ip
       self.sport  = sport
       self.dport  = dport
       now = _time.monotonic()
       self.start = self.last = self._last_pkt_ts = now


       self.fwd_pkts = self.bwd_pkts = 0
       self.fwd_bytes = self.bwd_bytes = 0
       self.pkt_lengths = []
       self.fwd_pkt_lengths = []
       self.bwd_pkt_lengths = []
       self.flow_iat = []
       self.fwd_iat  = []
       self.bwd_iat  = []


       self.flag_fin = self.flag_syn = self.flag_rst = 0
       self.flag_psh = self.flag_ack = self.flag_urg = 0
       self.flag_cwe = self.flag_ece = 0


       self.fwd_hdr_bytes = 0
       self.bwd_hdr_bytes = 0
       self.init_win_fwd = _CIC_IMPUTE["init_win_fwd"]
       self.init_win_bwd = _CIC_IMPUTE["init_win_bwd"]
       self._fwd_win_seen = False
       self._bwd_win_seen = False


   # ------------------------------------------------------------------
   def add_packet(self, pkt) -> None:
       now     = _time.monotonic()
       pkt_len = len(pkt)
       iat     = now - self._last_pkt_ts


       self.flow_iat.append(iat)
       self._last_pkt_ts = now
       self.last = now
       self.pkt_lengths.append(pkt_len)


       is_fwd = True
       if SCAPY_AVAILABLE and pkt.haslayer(IP):
           is_fwd = (pkt[IP].src == self.src_ip)


       if is_fwd:
           self.fwd_pkts  += 1
           self.fwd_bytes += pkt_len
           self.fwd_pkt_lengths.append(pkt_len)
           self.fwd_iat.append(iat)
           self.fwd_hdr_bytes += 20  # IP header baseline
           if SCAPY_AVAILABLE and pkt.haslayer(TCP):
               self.fwd_hdr_bytes += pkt[TCP].dataofs * 4 - 20
       else:
           self.bwd_pkts  += 1
           self.bwd_bytes += pkt_len
           self.bwd_pkt_lengths.append(pkt_len)
           self.bwd_iat.append(iat)
           self.bwd_hdr_bytes += 20


       if SCAPY_AVAILABLE and pkt.haslayer(TCP):
           f = pkt[TCP].flags
           self.flag_fin += int(bool(f & 0x01))
           self.flag_syn += int(bool(f & 0x02))
           self.flag_rst += int(bool(f & 0x04))
           self.flag_psh += int(bool(f & 0x08))
           self.flag_ack += int(bool(f & 0x10))
           self.flag_urg += int(bool(f & 0x20))
           self.flag_ece += int(bool(f & 0x40))
           self.flag_cwe += int(bool(f & 0x80))
           if is_fwd and not self._fwd_win_seen:
               self.init_win_fwd  = pkt[TCP].window
               self._fwd_win_seen = True
           elif not is_fwd and not self._bwd_win_seen:
               self.init_win_bwd  = pkt[TCP].window
               self._bwd_win_seen = True


   # ------------------------------------------------------------------
   @staticmethod
   def _arr(lst: list) -> np.ndarray:
       return np.asarray(lst, dtype=float) if lst else np.array([0.0])


   @staticmethod
   def _safe_std(arr: np.ndarray) -> float:
       return float(arr.std()) if len(arr) > 1 else 0.0


   def _duration(self) -> float:
       """Stabilised duration: at least _MIN_FLOW_SEC to prevent rate spikes."""
       return max(self.last - self.start, _MIN_FLOW_SEC)


   # ------------------------------------------------------------------
   def to_feature_vector(self, dataset_key: str, n_features: int) -> np.ndarray:
       """
       Returns ndarray of shape (1, n_features) with values placed at
       exactly the column positions expected by the trained scaler/model
       for `dataset_key`.  Uncomputable columns are imputed with dataset
       means (not zeros) to keep scaled values near z = 0.
       """
       dur    = self._duration()        # seconds, stabilised
       dur_us = dur * 1_000_000         # microseconds (CIC convention)

       # For per-second RATE features, use a floor of at least 1 second so a
       # single freshly-seen packet does not produce an absurd bytes/sec spike
       # (e.g. 1500 B / 0.01 s = 150 000 B/s) that pushes the scaled value far
       # out of the training distribution and triggers false positives.
       rate_dur = max(self.last - self.start, 1.0)


       tot_pkts  = self.fwd_pkts  + self.bwd_pkts
       tot_bytes = self.fwd_bytes + self.bwd_bytes


       pkt_arr = self._arr(self.pkt_lengths)
       fwd_arr = self._arr(self.fwd_pkt_lengths)
       bwd_arr = self._arr(self.bwd_pkt_lengths)
       iat_arr = self._arr(self.flow_iat) * 1e6   # -> microseconds
       fiat    = self._arr(self.fwd_iat)  * 1e6
       biat    = self._arr(self.bwd_iat)  * 1e6


       flow_bytes_s  = tot_bytes  / rate_dur
       flow_pkts_s   = tot_pkts   / rate_dur
       fwd_pkts_s    = self.fwd_pkts  / rate_dur
       bwd_pkts_s    = self.bwd_pkts  / rate_dur
       down_up_ratio = self.bwd_bytes / max(self.fwd_bytes, 1)


       pkt_mean = float(pkt_arr.mean())
       pkt_std  = self._safe_std(pkt_arr)
       pkt_var  = pkt_std ** 2


       cols = DATASET_COLS.get(dataset_key, _CIC2017_COLS)
       v    = np.zeros(max(n_features, len(cols)), dtype=float)


       # ── CIC-IDS-2017 (78 features) ─────────────────────────────────
       if dataset_key == "CIC-IDS-2017":
           v[0]  = self.dport
           v[1]  = dur_us
           v[2]  = self.fwd_pkts
           v[3]  = self.bwd_pkts
           v[4]  = self.fwd_bytes
           v[5]  = self.bwd_bytes
           v[6]  = float(fwd_arr.max())
           v[7]  = float(fwd_arr.min())
           v[8]  = float(fwd_arr.mean())
           v[9]  = self._safe_std(fwd_arr)
           v[10] = float(bwd_arr.max())
           v[11] = float(bwd_arr.min())
           v[12] = float(bwd_arr.mean())
           v[13] = self._safe_std(bwd_arr)
           v[14] = flow_bytes_s
           v[15] = flow_pkts_s
           v[16] = float(iat_arr.mean())
           v[17] = self._safe_std(iat_arr)
           v[18] = float(iat_arr.max())
           v[19] = float(iat_arr.min())
           v[20] = float(fiat.sum())
           v[21] = float(fiat.mean())
           v[22] = self._safe_std(fiat)
           v[23] = float(fiat.max())
           v[24] = float(fiat.min())
           v[25] = float(biat.sum())
           v[26] = float(biat.mean())
           v[27] = self._safe_std(biat)
           v[28] = float(biat.max())
           v[29] = float(biat.min())
           v[30] = self.flag_psh          # Fwd PSH Flags
           v[31] = 0                      # Bwd PSH Flags — not tracked separately
           v[32] = self.flag_urg          # Fwd URG Flags
           v[33] = 0                      # Bwd URG Flags
           v[34] = self.fwd_hdr_bytes
           v[35] = self.bwd_hdr_bytes
           v[36] = fwd_pkts_s
           v[37] = bwd_pkts_s
           v[38] = float(pkt_arr.min())
           v[39] = float(pkt_arr.max())
           v[40] = pkt_mean
           v[41] = pkt_std
           v[42] = pkt_var
           v[43] = self.flag_fin
           v[44] = self.flag_syn
           v[45] = self.flag_rst
           v[46] = self.flag_psh
           v[47] = self.flag_ack
           v[48] = self.flag_urg
           v[49] = self.flag_cwe
           v[50] = self.flag_ece
           v[51] = down_up_ratio
           v[52] = pkt_mean               # Average Packet Size
           v[53] = float(fwd_arr.mean())  # Avg Fwd Segment Size
           v[54] = float(bwd_arr.mean())  # Avg Bwd Segment Size
           v[55] = self.fwd_hdr_bytes     # Fwd Header Length.1 (duplicate col)
           # v[56..61] bulk features — imputed 0 (benign baseline)
           v[62] = self.fwd_pkts          # Subflow Fwd Packets
           v[63] = self.fwd_bytes         # Subflow Fwd Bytes
           v[64] = self.bwd_pkts          # Subflow Bwd Packets
           v[65] = self.bwd_bytes         # Subflow Bwd Bytes
           v[66] = self.init_win_fwd
           v[67] = self.init_win_bwd
           v[68] = max(self.fwd_pkts - 1, 0)  # act_data_pkt_fwd
           v[69] = 20                          # min_seg_size_forward (IP header)
           # v[70..77] Active/Idle — imputed 0


       # ── CIC-IDS-2018 (exact 47 columns from real CSV) ──────────────
       # Verified against live_traffic_sample.csv: this pipeline dropped many
       # CICFlowMeter columns and has NO Destination Port / Protocol. Order:
       # Flow Duration ... Idle Std (47 features, Label removed).
       elif dataset_key == "CIC-IDS-2018":
           _cic2018_vals = [
               dur_us,                  # 0  Flow Duration
               self.fwd_pkts,           # 1  Total Fwd Packets
               self.bwd_pkts,           # 2  Total Backward Packets
               float(fwd_arr.max()),    # 3  Fwd Packet Length Max
               float(fwd_arr.min()),    # 4  Fwd Packet Length Min
               float(fwd_arr.mean()),   # 5  Fwd Packet Length Mean
               self._safe_std(fwd_arr), # 6  Fwd Packet Length Std
               float(bwd_arr.max()),    # 7  Bwd Packet Length Max
               float(bwd_arr.min()),    # 8  Bwd Packet Length Min
               float(bwd_arr.mean()),   # 9  Bwd Packet Length Mean
               self._safe_std(bwd_arr), # 10 Bwd Packet Length Std
               flow_bytes_s,            # 11 Flow Bytes/s
               flow_pkts_s,             # 12 Flow Packets/s
               float(iat_arr.mean()),   # 13 Flow IAT Mean
               self._safe_std(iat_arr), # 14 Flow IAT Std
               float(iat_arr.max()),    # 15 Flow IAT Max
               float(iat_arr.min()),    # 16 Flow IAT Min
               float(fiat.mean()),      # 17 Fwd IAT Mean
               self._safe_std(fiat),    # 18 Fwd IAT Std
               float(biat.sum()),       # 19 Bwd IAT Total
               float(biat.mean()),      # 20 Bwd IAT Mean
               self._safe_std(biat),    # 21 Bwd IAT Std
               float(biat.max()),       # 22 Bwd IAT Max
               float(biat.min()),       # 23 Bwd IAT Min
               self.flag_psh,           # 24 Fwd PSH Flags
               self.flag_urg,           # 25 Fwd URG Flags
               bwd_pkts_s,              # 26 Bwd Packets/s
               float(pkt_arr.min()),    # 27 Packet Length Min
               float(pkt_arr.max()),    # 28 Packet Length Max
               pkt_mean,                # 29 Packet Length Mean
               pkt_std,                 # 30 Packet Length Std
               pkt_var,                 # 31 Packet Length Variance
               self.flag_fin,           # 32 FIN Flag Count
               self.flag_rst,           # 33 RST Flag Count
               self.flag_psh,           # 34 PSH Flag Count
               self.flag_ack,           # 35 ACK Flag Count
               self.flag_urg,           # 36 URG Flag Count
               down_up_ratio,           # 37 Down/Up Ratio
               self.init_win_fwd,       # 38 Init Fwd Win Bytes
               self.init_win_bwd,       # 39 Init Bwd Win Bytes
               20,                      # 40 Fwd Seg Size Min (IP header baseline)
               0,                       # 41 Active Mean  (imputed)
               0,                       # 42 Active Std
               0,                       # 43 Active Max
               0,                       # 44 Active Min
               0,                       # 45 Idle Mean
               0,                       # 46 Idle Std
           ]
           for i, val in enumerate(_cic2018_vals):
               if i < n_features:
                   v[i] = float(val)


       # ── ToN-IoT-v3  (NF-ToN-IoT-v3, exact 46 NetFlow columns) ──────
       # Column order verified against the real live_traffic_sample.csv header.
       # Training kept all 46 numeric NetFlow fields (Label dropped) + MinMaxScaler.
       # Fields that cannot be reconstructed from raw sniffed packets
       # (retransmissions, throughput aggregates, DNS/FTP, per-direction IAT
       # stddev, min/max TTL) are imputed; ToN-IoT live-capture is therefore
       # approximate by nature, but the layout now matches the model exactly.
       elif dataset_key == "ToN-IoT-v3":
           tcp_flags_or = (
               (0x01 if self.flag_fin else 0) | (0x02 if self.flag_syn else 0) |
               (0x04 if self.flag_rst else 0) | (0x08 if self.flag_psh else 0) |
               (0x10 if self.flag_ack else 0) | (0x20 if self.flag_urg else 0)
           )
           _l7 = {80: 7, 443: 91, 53: 5, 22: 92, 21: 1, 25: 3}.get(self.dport, 0)
           _dur_s   = max(self.last - self.start, 0.0)
           _icmp_t  = 1 if self.proto == 1 else 0   # ICMP only

           # forward/backward IAT in microseconds (already *1e6 above)
           _fi_min = float(fiat.min()); _fi_max = float(fiat.max())
           _fi_avg = float(fiat.mean()); _fi_std = self._safe_std(fiat)
           _bi_min = float(biat.min()); _bi_max = float(biat.max())
           _bi_avg = float(biat.mean()); _bi_std = self._safe_std(biat)

           _ton_vals = [
               self.proto,            # 0  PROTOCOL
               _l7,                   # 1  L7_PROTO
               self.fwd_bytes,        # 2  IN_BYTES
               self.fwd_pkts,         # 3  IN_PKTS
               self.bwd_bytes,        # 4  OUT_BYTES
               self.bwd_pkts,         # 5  OUT_PKTS
               tcp_flags_or,          # 6  TCP_FLAGS
               tcp_flags_or,          # 7  CLIENT_TCP_FLAGS
               tcp_flags_or,          # 8  SERVER_TCP_FLAGS
               _dur_s,                # 9  DURATION_IN  (seconds)
               _dur_s,                # 10 DURATION_OUT
               64,                    # 11 MIN_TTL  (imputed typical)
               64,                    # 12 MAX_TTL
               float(pkt_arr.max()),  # 13 LONGEST_FLOW_PKT
               float(pkt_arr.min()),  # 14 SHORTEST_FLOW_PKT
               float(pkt_arr.min()),  # 15 MIN_IP_PKT_LEN
               float(pkt_arr.max()),  # 16 MAX_IP_PKT_LEN
               flow_bytes_s,          # 17 SRC_TO_DST_SECOND_BYTES (approx)
               flow_bytes_s,          # 18 DST_TO_SRC_SECOND_BYTES (approx)
               0,                     # 19 RETRANSMITTED_IN_BYTES  (can't sniff)
               0,                     # 20 RETRANSMITTED_IN_PKTS
               0,                     # 21 RETRANSMITTED_OUT_BYTES
               0,                     # 22 RETRANSMITTED_OUT_PKTS
               flow_pkts_s,           # 23 SRC_TO_DST_AVG_THROUGHPUT (approx)
               flow_pkts_s,           # 24 DST_TO_SRC_AVG_THROUGHPUT
               self.fwd_pkts,         # 25 NUM_PKTS_UP_TO_128_BYTES (approx)
               0,                     # 26 NUM_PKTS_128_TO_256_BYTES
               0,                     # 27 NUM_PKTS_256_TO_512_BYTES
               0,                     # 28 NUM_PKTS_512_TO_1024_BYTES
               0,                     # 29 NUM_PKTS_1024_TO_1514_BYTES
               self.init_win_fwd,     # 30 TCP_WIN_MAX_IN
               self.init_win_bwd,     # 31 TCP_WIN_MAX_OUT
               _icmp_t,               # 32 ICMP_TYPE
               _icmp_t,               # 33 ICMP_IPV4_TYPE
               0,                     # 34 DNS_QUERY_ID    (no L7 parse)
               0,                     # 35 DNS_QUERY_TYPE
               0,                     # 36 DNS_TTL_ANSWER
               0,                     # 37 FTP_COMMAND_RET_CODE
               _fi_min,               # 38 SRC_TO_DST_IAT_MIN
               _fi_max,               # 39 SRC_TO_DST_IAT_MAX
               _fi_avg,               # 40 SRC_TO_DST_IAT_AVG
               _fi_std,               # 41 SRC_TO_DST_IAT_STDDEV
               _bi_min,               # 42 DST_TO_SRC_IAT_MIN
               _bi_max,               # 43 DST_TO_SRC_IAT_MAX
               _bi_avg,               # 44 DST_TO_SRC_IAT_AVG
               _bi_std,               # 45 DST_TO_SRC_IAT_STDDEV
           ]
           for i, val in enumerate(_ton_vals):
               if i < n_features:
                   v[i] = float(val)


       # ── Cross-Domain  (RobustScaler, reduced aligned feature set) ──
       # Cross-validation preprocess.py does: align CIC<->ToN common columns,
       # drop leakage, VarianceThreshold + correlation(>0.85) drop, RobustScaler,
       # binary labels. The SURVIVING feature set/order is data-dependent and
       # cannot be reproduced from code alone. We fill the most likely aligned
       # numeric fields; verify against the real scaler with inspect_features.py.
       elif dataset_key == "Cross-Domain":
           _cross_vals = [
               self.proto,                       # protocol
               (self.last - self.start) * 1000., # flow_duration_milliseconds
               self.fwd_pkts,                    # in_pkts
               self.bwd_pkts,                    # out_pkts
               self.fwd_bytes,                   # in_bytes
               self.bwd_bytes,                   # out_bytes
               float(fwd_arr.max()),             # longest_flow_pkt
               float(fwd_arr.min()),             # shortest_flow_pkt
               flow_bytes_s,                     # bytes/sec
               flow_pkts_s,                      # pkts/sec
               pkt_mean,                         # mean pkt size
               down_up_ratio,                    # down/up ratio
           ]
           for i, val in enumerate(_cross_vals):
               if i < n_features:
                   v[i] = float(val)


       # Trim or zero-pad to exact n_features required by scaler
       if len(v) >= n_features:
           return v[:n_features].reshape(1, -1)
       padded = np.zeros(n_features, dtype=float)
       padded[:len(v)] = v
       return padded.reshape(1, -1)

# FlowTracker — keyed flow table (5-tuple, bidirectional)
class FlowTracker:
   """
   Maintains a 5-tuple flow table.  Each packet is assigned to an existing
   flow or starts a new one.  Stale flows are evicted after FLOW_TIMEOUT.
   Must be stored in st.session_state so it persists across Streamlit reruns.
   """


   def __init__(self):
       self._flows: dict = {}
       self._pkt_since_evict = 0
       self._EVICT_EVERY = 50   # run stale-eviction at most once per 50 packets


   def add_packet(self, pkt) -> _FlowRecord:
       key = self._flow_key(pkt)
       if key not in self._flows:
           proto, src_ip, dst_ip, sport, dport = self._pkt_meta(pkt)
           self._flows[key] = _FlowRecord(proto, src_ip, dst_ip, sport, dport)
       self._flows[key].add_packet(pkt)
       # Eviction is O(n) over the flow table; running it every packet is
       # wasteful. Amortise it to once every _EVICT_EVERY packets.
       self._pkt_since_evict += 1
       if self._pkt_since_evict >= self._EVICT_EVERY:
           self._evict_stale()
           self._pkt_since_evict = 0
       return self._flows[key]


   def get_features(self, pkt, dataset_key: str, n_features: int) -> np.ndarray:
       flow = self.add_packet(pkt)
       return flow.to_feature_vector(dataset_key, n_features)


   def size(self) -> int:
       return len(self._flows)


   def reset(self) -> None:
       self._flows.clear()
       self._pkt_since_evict = 0


   @staticmethod
   def _pkt_meta(pkt):
       proto = 0; src_ip = "0.0.0.0"; dst_ip = "0.0.0.0"; sport = 0; dport = 0
       if SCAPY_AVAILABLE and pkt.haslayer(IP):
           proto  = pkt[IP].proto
           src_ip = pkt[IP].src
           dst_ip = pkt[IP].dst
           if pkt.haslayer(TCP):
               sport = pkt[TCP].sport; dport = pkt[TCP].dport
           elif pkt.haslayer(UDP):
               sport = pkt[UDP].sport; dport = pkt[UDP].dport
       return proto, src_ip, dst_ip, sport, dport


   def _flow_key(self, pkt) -> tuple:
       proto, src_ip, dst_ip, sport, dport = self._pkt_meta(pkt)
       # Normalise bidirectional: A->B and B->A map to the same key
       if (src_ip, sport) > (dst_ip, dport):
           src_ip, dst_ip = dst_ip, src_ip
           sport,  dport  = dport,  sport
       return (proto, src_ip, dst_ip, sport, dport)


   def _evict_stale(self) -> None:
       now   = _time.monotonic()
       stale = [k for k, f in self._flows.items()
                if now - f.last > _FlowRecord.FLOW_TIMEOUT]
       for k in stale:
           del self._flows[k]

# PortScanDetector — cross-flow heuristic (nmap / hping3)
# WHY THIS IS NEEDED:
#   nmap SYN scan emits one SYN packet per destination port.
#   Each (src->dst:port) becomes a separate flow with only one packet.
#   FlowTracker + ML model never accumulates enough per-flow statistics,
#   so the model almost always predicts "Normal" for individual SYN pkts.
#
#   Solution: track cross-flow behaviour per source IP.
#   If a single src_ip contacts >= PORT_SCAN_THRESHOLD distinct dst_ports
#   within PORT_SCAN_WINDOW seconds, declare "Port-Scan" regardless of
#   the per-flow ML prediction — mirroring Snort SID:1228 / Suricata ET.


class PortScanDetector:
   PORT_SCAN_WINDOW    = 10.0  # sliding observation window (seconds)
   PORT_SCAN_THRESHOLD = 15    # distinct weighted ports to trigger alert
   SYN_ONLY_WEIGHT     = 2     # SYN-only flows count double (nmap default)


   def __init__(self):
       # src_ip -> list of (timestamp, dst_port, is_syn_only)
       self._table: dict = {}


   def observe(self, src_ip: str, dst_port: int, flags: int) -> bool:
       """
       Record a packet from src_ip to dst_port.
       flags: TCP flags integer (0 if non-TCP).
       Returns True when src_ip is classified as a port scanner.
       """
       now    = _time.monotonic()
       is_syn = bool(flags & 0x02) and not bool(flags & 0x10)  # SYN set, ACK not


       self._table.setdefault(src_ip, [])
       self._table[src_ip].append((now, dst_port, is_syn))
       # Prune entries outside the sliding window
       self._table[src_ip] = [
           (t, p, s) for (t, p, s) in self._table[src_ip]
           if now - t <= self.PORT_SCAN_WINDOW
       ]


       port_set = set()
       weighted = 0
       for (_, p, s) in self._table[src_ip]:
           if p not in port_set:
               port_set.add(p)
               weighted += self.SYN_ONLY_WEIGHT if s else 1


       return weighted >= self.PORT_SCAN_THRESHOLD


   def reset(self) -> None:
       self._table.clear()


# Network interface resolver (Windows friendly names)
def get_friendly_interfaces() -> list:
   """
   Returns list of (friendly_label, scapy_iface_id) tuples.
   Resolves GUID-style Windows interface names to human-readable labels
   using three methods in order:
       1. scapy.arch.windows.get_windows_if_list()
       2. PowerShell Get-NetAdapter
       3. WMI fallback
   On Linux/macOS the raw names (eth0, wlan0) are already readable.
   """
   import subprocess, json, re as _re


   _PRIORITY = ["wi-fi", "wifi", "wireless", "wlan", "realtek", "intel",
                "ethernet", "local area connection"]
   _EXCLUDE  = ["loopback", "wan miniport", "wi-fi direct",
                "bluetooth", "isatap", "teredo", "6to4",
                "npcap loopback", "pseudo", "kernel debug",
                "virtualbox", "vmware", "vbox", "virtual"]

   guid_map = {}

   if sys.platform == "win32":
       # Method 1: Scapy's own Windows resolver (fastest)
       try:
           from scapy.arch.windows import get_windows_if_list as _scapy_wif
           for iface in _scapy_wif():
               guid = (iface.get("guid") or "").strip("{}").upper()
               desc = (iface.get("description") or "").strip()
               name = (iface.get("name") or "").strip()
               if guid and (desc or name):
                   guid_map[guid] = {"desc": desc or name, "name": name, "status": "Up"}
       except Exception:
           pass


       # Method 2: PowerShell fallback
       if not guid_map:
           try:
               ps_cmd = (
                   "Get-NetAdapter -IncludeHidden | "
                   "Select-Object Name,InterfaceDescription,InterfaceGuid,Status | "
                   "ConvertTo-Json -Compress"
               )
               res = subprocess.run(
                   ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_cmd],
                   capture_output=True, text=True, timeout=8,
               )
               adapters = json.loads(res.stdout or "[]")
               if isinstance(adapters, dict):
                   adapters = [adapters]
               for a in adapters:
                   guid   = (a.get("InterfaceGuid") or "").strip("{}").upper()
                   desc   = (a.get("InterfaceDescription") or "").strip()
                   name   = (a.get("Name") or "").strip()
                   status = (a.get("Status") or "").strip()
                   if guid and guid not in guid_map:
                       guid_map[guid] = {"desc": desc or name, "name": name, "status": status}
           except Exception:
               pass


       # Method 3: WMI fallback (optional dependency)
       if not guid_map:
           try:
               import wmi as _wmi
               for nic in _wmi.WMI().Win32_NetworkAdapter():
                   guid = (nic.GUID or "").strip("{}").upper()
                   name = (nic.Name or "").strip()
                   if guid and name:
                       guid_map[guid] = {"desc": name, "name": name, "status": "Unknown"}
           except Exception:
               pass


   raw_list = []
   try:
       raw_list = get_if_list() if SCAPY_AVAILABLE else []
   except Exception:
       pass
   if not raw_list:
       raw_list = ["Wi-Fi", "Ethernet"] if sys.platform == "win32" else ["eth0", "wlan0"]


   pairs = []
   for raw in raw_list:
       m    = _re.search(r"[{]([A-F0-9\-]+)[}]", raw.upper())
       info = guid_map.get(m.group(1)) if m else None


       if info:
           desc   = info["desc"]
           status = info.get("status", "")
           desc_l = desc.lower()
           if any(kw in desc_l for kw in _EXCLUDE):
               continue
           label = f"{desc} [{status}]" if status and status not in ("Up", "Unknown", "") else desc
       else:
           desc_l = raw.lower()
           if any(kw in desc_l for kw in _EXCLUDE):
               continue
           if raw.upper().startswith(r"\DEVICE\NPF_"):
               label = f"Adapter {raw[-8:].strip('{}')}"
           else:
               label = raw


       priority = next((i for i, kw in enumerate(_PRIORITY) if kw in desc_l), len(_PRIORITY))
       pairs.append((priority, label, raw))


   pairs.sort(key=lambda x: (x[0], x[1]))
   if not pairs:
       pairs = [(0, "Wi-Fi (fallback)", "Wi-Fi")]
   return [(lbl, raw) for (_, lbl, raw) in pairs]


# Model / scaler loaders
def load_framework_assets(dataset_dir: str, model_filename: str):
   """
   Load model, scaler, and target_names from dataset_dir.
   Returns (model, scaler, target_names); each is None if the file is absent.
   NOTE: Call this inside @st.cache_resource in app.py to avoid reloading.
   """
   model_path  = os.path.join(dataset_dir, model_filename)
   scaler_path = os.path.join(dataset_dir, "scaler.pkl")
   target_path = os.path.join(dataset_dir, "target_names.pkl")
   model = scaler = target_names = None


   if os.path.exists(model_path):
       if model_filename.endswith(".h5"):
           import tensorflow as tf
           model = tf.keras.models.load_model(model_path, compile=False)
       elif model_filename.endswith(".pkl"):
           model = joblib.load(model_path)


   if os.path.exists(scaler_path):
       scaler = joblib.load(scaler_path)
   if os.path.exists(target_path):
       target_names = joblib.load(target_path)


   return model, scaler, target_names

def load_csv_source(csv_path: str):
   """Load live_traffic_sample.csv; returns DataFrame or None."""
   if os.path.exists(csv_path):
       return pd.read_csv(csv_path)
   return None

def get_feature_count(active_scaler, active_model, dataset_choice: str) -> int:
   """
   Dynamically infer the required feature count from loaded assets.
   Priority: scaler.n_features_in_ -> model.input_shape -> dataset default.
   """
   if active_scaler and hasattr(active_scaler, "n_features_in_"):
       return int(active_scaler.n_features_in_)
   if active_model is not None:
       try:
           shape = active_model.input_shape
           if isinstance(shape, list):
               shape = shape[0]
           return int(shape[-1])
       except Exception:
           pass
   # Fallback only — in practice scaler.n_features_in_ above always wins.
   # ToN-IoT-v3 here is the NetFlow numeric layout (~34), NOT the old ~470
   # one-hot count. Cross-Domain's true size is data-dependent (reduced set).
   _defaults = {
       "CIC-IDS-2017": len(_CIC2017_COLS),   # 78
       "CIC-IDS-2018": 47,                    # verified from real CSV (reduced set)
       "ToN-IoT-v3":   46,                    # NF-ToN-IoT-v3 (verified from CSV)
       "Cross-Domain": 12,                    # reduced aligned set (approx)
   }
   return _defaults.get(dataset_choice, 46)

# Utility helpers
def clean_label(label_str: str) -> str:
   """Normalise a raw class label to a clean human-readable string."""
   if label_str is None:
       return "Unknown"
   s = re.sub(r'[^\x00-\x7F]+', '-', str(label_str))
   s = s.replace('--', '-').strip('-').strip()
   if s.upper() in {"0", "0.0", "NORMAL", "BENIGN", "NONE", "NAN"}:
       return "Normal" if s.upper() in {"0", "0.0", "NORMAL", "BENIGN"} else "Unknown"
   if s in {"1", "1.0"}:
       return "Attack"
   return s if s else "Unknown"


def safe_int(val, default: int = 0) -> int:
   try:
       return int(float(val))
   except Exception:
       return default


def resolve_dataset_dir(monitoring_mode: str, balancing_choice: str,
                       dataset_choice: str) -> str:
   """Return the absolute path to the model/scaler/csv directory."""
   if monitoring_mode == "Cross-Domain (Robustness Test)":
       return os.path.join(BASE_DIR, "Cross_Validation")
   bal = BALANCING_FOLDER_MAP.get(balancing_choice, "")
   ds  = DATASET_FOLDER_MAP.get(dataset_choice, "")
   return os.path.join(BASE_DIR, bal, ds)


def resolve_model_file(monitoring_mode: str, model_choice: str) -> str:
   """Return the model filename for the given mode + model selection."""
   if monitoring_mode == "Cross-Domain (Robustness Test)" and model_choice == "CNN+Transformer":
       return "CNN_Transformer_model.h5"
   return MODEL_FILE_MAP.get(model_choice, "")