#!/usr/bin/env python3
"""Read-only final24 raw-score audit, previous32 hash continuity, and independent paired gates.

Run by stdin from source_teacher_v2_delivery using the live Python environment.
No inference, physics, hardware or primary artifact writes.
"""

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from phantom.sim.policy_metrics import evaluate_policy_trace
from tools.sim.analyze_policy_campaign import (
    condition_scene,
    read_rows,
    runtime_audit,
    scene_mismatches,
)
from tools.sim.select_teacher_candidate import evaluate_selection, trial_evidence

BASE = Path("/home/physicalai/phantom-icra-2027/sim/waffles")
ROOT = BASE / "runs/teacher_robustness_v2_delivery"
CONFIRM_SHA = "8242935967b8d16616759fd3282834c346f16f51f779f2a2dab64ecdb20298fe"
SCREEN_SELECTION_SHA = (
    "3417ffb569caee0d0b7c0a07fac0e667f968dc19be6961cb0b970cd2f6a7a63f"
)
PRIOR_SCREEN_AUDIT_SHA = (
    "6eeb4c6a9bda658768f2b5b1528553d764589f8f62ca4c23f1a022eda5ff2bbc"
)
PRIOR_SCREEN_RAW_HASHES = {
    "fta1500_nfe1_k4__start_1787395928__seed903101": {
        "sim_trace.npz": "4da663c5f912fe092cd05b037694dd8a8cb5ab902bbc82cd9950dc5a4acb49ab",
        "execution_trace.jsonl": "a9c930e4538fa81088a689094b71cff81eda0612174f126ef12dda3bac2e7d29",
        "planner_trace.json": "36e1ba384e43998402c5d1a0b23fc1b935f62a8ba082003d0dfb9ac4a5b0905c",
        "run.json": "2b47fed4540e0924cd961582638ced27fddc9e795921e003b70d0103aa5093a1",
        "effective_config.json": "58ef867d5f9f74131a2f65e37e529c8b7c53a006679fe5072c898a58af86c7fe",
        "case.json": "1ab055c75cc6dcc6ec1a68e15d385512d6735b5bed25da422c7905bf857eab24",
    },
    "fta1500_nfe1_k4__start_1787395928__seed903102": {
        "sim_trace.npz": "8b48ec21db154640ee8bd16411b3ec4f1577a489280fbebf1b7afec93b261417",
        "execution_trace.jsonl": "7991f49f7b984cb846b67a04161be38b285dd83469d2c063088410f6e539d327",
        "planner_trace.json": "e2431fc7ae47d6ced46f14b8c803357985cd7f7698b709d781278bd15d1981fd",
        "run.json": "8e5948f9d6021296b25c6294a40aa134c6a9ee7771fc46197728581380fa0d1c",
        "effective_config.json": "58ef867d5f9f74131a2f65e37e529c8b7c53a006679fe5072c898a58af86c7fe",
        "case.json": "07faee8b2844c89dadc9283d29b28992c9c46a2776ef8e7d250849ca823d8fd3",
    },
    "fta1500_nfe1_k4__start_1787395963__seed903101": {
        "sim_trace.npz": "df5e8037e262de703149bd9d0a518ea0b3dabf1f700b685e6f32e5cf79259685",
        "execution_trace.jsonl": "3d0b9ab7e40b32daeeb75ddeefb1b48b64c2fd93b15609f2dadff2e3e68f3145",
        "planner_trace.json": "28e7d247865db569b5d19e4e1e9eeed7818c1f4f54eec24f884fd4846023b204",
        "run.json": "f027169526eac38bc7817360e1613601c44f01c265154355b13c51d30c70a0a1",
        "effective_config.json": "58ef867d5f9f74131a2f65e37e529c8b7c53a006679fe5072c898a58af86c7fe",
        "case.json": "49e8a7a981b4294926c97f8ede4ce174be0852f74e9b4502e3e14573fc82a3db",
    },
    "fta1500_nfe1_k4__start_1787395963__seed903102": {
        "sim_trace.npz": "922a616272cc850a03c72a8896574857d1d15d2d6bcd1cc3974717512cc7b54b",
        "execution_trace.jsonl": "9abc958e9fd38d473218dbbf43254e0e4861f0b9f56b8275c42fc94158cf2231",
        "planner_trace.json": "d1c6f6ad7543c58c87c76f050ea3fb11b0b3f6c93fdd6dc871b5b9bc1cfa59d0",
        "run.json": "e0ff3b662d96391b353d4b0290dfcd98e0aa9c8cb88cec08b6685cb8173fdb59",
        "effective_config.json": "58ef867d5f9f74131a2f65e37e529c8b7c53a006679fe5072c898a58af86c7fe",
        "case.json": "f2baa652861bd194229fb825ef522549b3bbd28a4b94cc1257747fdeaae3000e",
    },
    "fta1500_nfe1_k4__start_1787396028__seed903101": {
        "sim_trace.npz": "df5c5e6b171dce3d709779334b047db4beffa8802a085c82a5e8b946f3ce765f",
        "execution_trace.jsonl": "27d6ceee2cdf22d241b74a495f1ced8c58aeea7c156aa3d598d9e7dc7029d690",
        "planner_trace.json": "a1fac5286d7a534e08d10a637190ccdeb08199d82424c606631a53420023bb3d",
        "run.json": "0c0a49c5da0e1289443bda326cf128f8885d7c997aa5f3ca2d6c408f320e5cc7",
        "effective_config.json": "58ef867d5f9f74131a2f65e37e529c8b7c53a006679fe5072c898a58af86c7fe",
        "case.json": "6da1285a8c4459b2ec89a27f5559e4ca45603eb8fee653ccb604fba9a262add9",
    },
    "fta1500_nfe1_k4__start_1787396028__seed903102": {
        "sim_trace.npz": "81ef4d7b3a086cf7dcd9dc715ed6a1726a3412795d38bc00c025b21f7b573a8b",
        "execution_trace.jsonl": "ad3046a7b8377367842bbf79d7b39614716473cc6f38b74b889a142582804f1a",
        "planner_trace.json": "30d224982952e1e849c6585de8721b8becde5ec2046ac70103d9bfb7c1771672",
        "run.json": "47c692c9403af4fbade7aca7ea923cbfb2f2bc8cb41b9f3e6da0e8afaf5bbf1f",
        "effective_config.json": "58ef867d5f9f74131a2f65e37e529c8b7c53a006679fe5072c898a58af86c7fe",
        "case.json": "ee9c50bab8effdf16db67e92a53f888dda035c8deff5246a9d3e275756d7f814",
    },
    "fta1500_nfe1_k4__start_1787396273__seed903101": {
        "sim_trace.npz": "009a50cdf6f889e4598f0dddd0503e15b0406e88f49b87afc3619cd153bd50ad",
        "execution_trace.jsonl": "8ea310d5a4f55940a0f176e8fb0b80b447ab8aa4e272990329764c5240c4ec68",
        "planner_trace.json": "3f164848a3d36f39eae89e2e8ea7d18c6feef930ce08de1b7831da54516e8397",
        "run.json": "21ad781416adb78d7e473f9a92cff8d826a078e1c16da75552fcdbe87c29e69d",
        "effective_config.json": "58ef867d5f9f74131a2f65e37e529c8b7c53a006679fe5072c898a58af86c7fe",
        "case.json": "17b44480820c62c98186ab457f12de421b8ab987b65e557bb9d29c76e55d02c4",
    },
    "fta1500_nfe1_k4__start_1787396273__seed903102": {
        "sim_trace.npz": "b99e5a9a5303ae35f355e1df318ed9e2c378717b68f6a1217733f81d19f638b8",
        "execution_trace.jsonl": "a38d75cb042d469070229a9ca0d430954124dc18283b412afbb059ac84742ed5",
        "planner_trace.json": "c142fd93391ba7ba65096cff11dba56b978215b41522a197eb9729c2613719ac",
        "run.json": "04d9b95ce4ca5f1247ff149026cb9046502e3939a6b2905f8d06e9ba3c0ebcf9",
        "effective_config.json": "58ef867d5f9f74131a2f65e37e529c8b7c53a006679fe5072c898a58af86c7fe",
        "case.json": "bfbd271e8006b190ad56a25c9db0feef3213880475379e935f5bc8e2e53cabab",
    },
    "fta1500_nfe5_k1__start_1787395928__seed903101": {
        "sim_trace.npz": "fff0ca61e0e6eb13392f49a69ba7c01c52e01ae51298015fcc21274776dd6e42",
        "execution_trace.jsonl": "2c0943296cf8f7529c9b395e6ee5535adc6609a249e571a37c279a23dd786c2a",
        "planner_trace.json": "ba0064bfaef8aba1794f0aa8ec888131c08e085e2b018dc9c891526c6f2ac9e4",
        "run.json": "4a131976613f7af6eb1962c7ae6664b8ee09b12f3a92def9f31a0e1d138e96aa",
        "effective_config.json": "58ef867d5f9f74131a2f65e37e529c8b7c53a006679fe5072c898a58af86c7fe",
        "case.json": "dd53671cf6339433ba0f2b7d06c7992991e555995f2aa7abaf1ef4a553438067",
    },
    "fta1500_nfe5_k1__start_1787395928__seed903102": {
        "sim_trace.npz": "1611fb3cb3da4f93f21d8aadeb4e76ed9c1b6a1bc1be6b99df5ec4757ee5ebde",
        "execution_trace.jsonl": "304d61bc449b5e17673d112ffd6cd852bf293eb96364df296070b5a364996b8e",
        "planner_trace.json": "10d08e3dea538a3b7f1d5295d2cdd658cfc31b86660249d48ae12eb42da6d52d",
        "run.json": "8d8bd14cc588a56bc619dfd29ebf35cc0e4aab63e9bc2082304fc4b2548c0854",
        "effective_config.json": "58ef867d5f9f74131a2f65e37e529c8b7c53a006679fe5072c898a58af86c7fe",
        "case.json": "1f693d3c6116cc6563b8e2b4c4dddee0517535d4bd2fee5263322f7d23c10a4d",
    },
    "fta1500_nfe5_k1__start_1787395963__seed903101": {
        "sim_trace.npz": "0be79fc009199333a4c5605aa9bbd6e534efeea10c1a6204bd929e767f7cfc20",
        "execution_trace.jsonl": "84ff162e7458d4424b3ae33ef4e458d7d0db7ce0c40185f160ed7e5b95538d2e",
        "planner_trace.json": "d86d8e9c9b96fe1f349bbbdf772bc9a3b708389de4487e6e0ba0742ac338ad64",
        "run.json": "170b2209cd0c5756b9ddb6dd2c8762248690b1a6a0f82a03e672b46d7afe4ef2",
        "effective_config.json": "58ef867d5f9f74131a2f65e37e529c8b7c53a006679fe5072c898a58af86c7fe",
        "case.json": "a86e90edbb96d4c834859690005cd620fa513120c5d0080e8fb0b1cc0a845934",
    },
    "fta1500_nfe5_k1__start_1787395963__seed903102": {
        "sim_trace.npz": "de98d0abff2954f8d3cef3d8974a968d7807b76c55de982b56a503054153c503",
        "execution_trace.jsonl": "45e3c7b0fcda3eb1f61f61b121f364005cb1d0507938584767ad6526b97537f6",
        "planner_trace.json": "89e4baeb7b3fffc66ec08a523f7400265c63e605c2a8d3acfbeb3b31cd6ae752",
        "run.json": "7b1db819d767d4180edd2a7367e1e821b9a5739e4254a86cc90cc3aa368d53d4",
        "effective_config.json": "58ef867d5f9f74131a2f65e37e529c8b7c53a006679fe5072c898a58af86c7fe",
        "case.json": "a77d4f6af0a8eee2012fa0284924d68ff19d8f153347cb958237a6d25495403f",
    },
    "fta1500_nfe5_k1__start_1787396028__seed903101": {
        "sim_trace.npz": "bba1d10313e3064525f26374bfccbc4e0e70d9162e3ce834fc6a0f21be6e72e7",
        "execution_trace.jsonl": "8eef090a1987e33a4c347f934346c47995268142868e3de9b7ab6ff700fe10b9",
        "planner_trace.json": "2ac9d1c1fa6d182098604047b0d387cafb960d74406341982de0f776bdd51dc2",
        "run.json": "4209c84137b5b938b9f2a62e0a8d9cca4df216e836d1c1ba313fcfa134652fda",
        "effective_config.json": "58ef867d5f9f74131a2f65e37e529c8b7c53a006679fe5072c898a58af86c7fe",
        "case.json": "d9a9be22b47f34415b05d9468890a8840f210b85df3c66e422bb544f2b090917",
    },
    "fta1500_nfe5_k1__start_1787396028__seed903102": {
        "sim_trace.npz": "2da4585f8554f8e8626746cdeffd3c03bf0877fe17ce82bad8967da5246ad278",
        "execution_trace.jsonl": "2d11add648834c4efd05fd7ba5fcfd5a3123acef905681acc4078eb6c2f4b269",
        "planner_trace.json": "b6b653d5cec406308909d2c1c2deb7ec0779d1f5db1013c3cac6d7dce69aefc8",
        "run.json": "03b1839aa61e69fadf9bfc08d2850e79656ae164582414a313a87fea0807dfca",
        "effective_config.json": "58ef867d5f9f74131a2f65e37e529c8b7c53a006679fe5072c898a58af86c7fe",
        "case.json": "1f26ee808a0013877bc8969ef147b1de9e7db09e7b23678faa661755a5f2304d",
    },
    "fta1500_nfe5_k1__start_1787396273__seed903101": {
        "sim_trace.npz": "ae7692de83c9eea7410b30f41ea28d04db3be33b26de3c07843b2bd2227849b9",
        "execution_trace.jsonl": "52b661f99119e7f32b0052fa01cfc677587af24755d16f5ac6642ed0a72a93a3",
        "planner_trace.json": "9d4113ef1c52544239758f89c009cea5a53a8cb43c895b9d05b9e311f0a66946",
        "run.json": "dbea0140330159ed7d5a2a79e0e6b3267f98cc967f5ea8b900785fd9f7498df1",
        "effective_config.json": "58ef867d5f9f74131a2f65e37e529c8b7c53a006679fe5072c898a58af86c7fe",
        "case.json": "1ff004d8d93997988df3dc51d783ecbd515fc0341ac4b2b26141a495607d1d83",
    },
    "fta1500_nfe5_k1__start_1787396273__seed903102": {
        "sim_trace.npz": "a9fe37620244bd4b053582b7c4a6bdbbef539b97b81a00a19cbb3413ff55e394",
        "execution_trace.jsonl": "a43bbab7ce021f39b2cc8a8b55431337ab377eb1a4882aed80f644e7b09ff397",
        "planner_trace.json": "9756ad0867182836c7f6bf37b9043c013bcfda2589b2c047fdec40a5f46ea32b",
        "run.json": "f74040be392a18b1cba314d8c026aca488b907d6d064d7f995b54f740a15fad4",
        "effective_config.json": "58ef867d5f9f74131a2f65e37e529c8b7c53a006679fe5072c898a58af86c7fe",
        "case.json": "911df66ed8d2444f2a5f0b4a663f015af4d3ca92ca0b1984ee5be8cc54e23fbe",
    },
    "fta3000_nfe1_k4__start_1787395928__seed903101": {
        "sim_trace.npz": "f9c1983fd676c14e57eb2c92c988252807a52e1021ca96474a6c6bf3a46e903b",
        "execution_trace.jsonl": "9e7404629c2188d7f9d753980b1c6afc3e9c9e08d08391ed944e8f9a88cc6113",
        "planner_trace.json": "0f6790d2b93b17bd419e5c0935370dd61b7a9f7c37611e914d5a91ff7990d261",
        "run.json": "1d83ddd4510eaa607945ef4c91e08ec342148db08ed59497787566efdcf97b18",
        "effective_config.json": "58ef867d5f9f74131a2f65e37e529c8b7c53a006679fe5072c898a58af86c7fe",
        "case.json": "aff77c78748cdc2dd69bc8ce495cd1925db107f3d72b351fa4d7e7bcc8382574",
    },
    "fta3000_nfe1_k4__start_1787395928__seed903102": {
        "sim_trace.npz": "b5014aadbb73650d6036b201e95f1c9f11e9136d5cde0250834800f35cffe92d",
        "execution_trace.jsonl": "7305c8728c543a046d714cd2eccf770709d86543c0f0875c6123b8308b5e9edf",
        "planner_trace.json": "5c335afd27d270abcf6bd479cd57ccecff6c7acc0582e8b78b0997c190c16d12",
        "run.json": "66d4c2c166dc46d095db36fb63305815f4f278e812e7250fcabead83583ffaa0",
        "effective_config.json": "58ef867d5f9f74131a2f65e37e529c8b7c53a006679fe5072c898a58af86c7fe",
        "case.json": "681e1ce37e1d037d3c05928ac6f783093a201c198f5d78f046148b25ec865509",
    },
    "fta3000_nfe1_k4__start_1787395963__seed903101": {
        "sim_trace.npz": "d5fefeefe18995516923cb3f33700c8adbd56ae2b449d806640b9720e1e0c328",
        "execution_trace.jsonl": "9a3e13bb9b1689a24aa7a24f9b9549d5283bd41e4e5a21f98321403f5727a5ad",
        "planner_trace.json": "4bfa6f67847c9e8cbf6da58ae1fec1acced224c487daa9730fba790692537460",
        "run.json": "ae06639015e574f1338a667e01096537463445c4a5b93f8848e1815e57501aba",
        "effective_config.json": "58ef867d5f9f74131a2f65e37e529c8b7c53a006679fe5072c898a58af86c7fe",
        "case.json": "35f2102ca6f43f37d43f0055537fd47883a5c59bf54e422f819206d5db810d93",
    },
    "fta3000_nfe1_k4__start_1787395963__seed903102": {
        "sim_trace.npz": "1bfa4c943a57c8d4446a2353476c112136d1b7a4eb097177729f3b2886586294",
        "execution_trace.jsonl": "a37d5ad6238ce6218e3783eaa475849ddb18f4e532f651de6e1df549c7e95ea9",
        "planner_trace.json": "7192e540ab8a109d8065317313af66d54d4b485de88237ea1210100784cb32dd",
        "run.json": "5b30115e868925930ec775b96bd6b3179d115bd98a26886be913c6783af22d03",
        "effective_config.json": "58ef867d5f9f74131a2f65e37e529c8b7c53a006679fe5072c898a58af86c7fe",
        "case.json": "d5a8147399d6881bb33d9a8dbec9115e749642573bd6eee06ebd24d6c2fec9f0",
    },
    "fta3000_nfe1_k4__start_1787396028__seed903101": {
        "sim_trace.npz": "716c25a7b55ab4aa4709cf7f4010d918910581709e9083a909819704f17b3387",
        "execution_trace.jsonl": "be7ecd9d8feda188c132cc93ef21bc810311b943ceded243913da443807c9468",
        "planner_trace.json": "b571b800da540ab62ce5bcf0c39ab64e015cf81e8c023d4fbb0b209cd76ab7b0",
        "run.json": "b0c9ddc359e205725c81914034923f3bf79edf7b580aa151627f7110848babdc",
        "effective_config.json": "58ef867d5f9f74131a2f65e37e529c8b7c53a006679fe5072c898a58af86c7fe",
        "case.json": "213d5bd58a2f0619317358734dec9a06e21a745f8bd99373d0cd64e1e2c9c9b8",
    },
    "fta3000_nfe1_k4__start_1787396028__seed903102": {
        "sim_trace.npz": "1b860bebdfd5282d7941962776690bdd054f71543611cc704ffadf1fee696856",
        "execution_trace.jsonl": "616b1ed8cac0960bf378e28cfb0f4cf970a28d7fda14b26685f2d127f8887773",
        "planner_trace.json": "557228b039ac3fd2431ded78dcf810392b2a7026f00b935c0b37928555b5327f",
        "run.json": "7e7d37bc5876fcf2904ff5a89c772464ca886e54eb4a662418010a92f1ff8d05",
        "effective_config.json": "58ef867d5f9f74131a2f65e37e529c8b7c53a006679fe5072c898a58af86c7fe",
        "case.json": "06ee6c74a69bb43e280f118dd1b884763fe3ab9d87fc8077cb7dc9a8ee010edf",
    },
    "fta3000_nfe1_k4__start_1787396273__seed903101": {
        "sim_trace.npz": "e5472b7383e95225ec2374a3e5a54fc653c327dd7865cd7e1bd7521a3f83e089",
        "execution_trace.jsonl": "9704d840b5b5cee5626e6dd2be740901d01187a330bc1908163dce2e5232ebc4",
        "planner_trace.json": "fdfd47b6c21965434c813888dd7598ddd5bf6d5cb7a4fa7ed884b3b4eab668cc",
        "run.json": "456e4655351924aa3f967b172506eebac0137fd55334afefe169aa3ef76e0ea0",
        "effective_config.json": "58ef867d5f9f74131a2f65e37e529c8b7c53a006679fe5072c898a58af86c7fe",
        "case.json": "f2f4aea77a54ae864b06efef8a8475cc0b2a1ab04ef873573f52694416958399",
    },
    "fta3000_nfe1_k4__start_1787396273__seed903102": {
        "sim_trace.npz": "89c153840d58b3e16d5661095cbdc49604a1d06995257bbdedbf181297119515",
        "execution_trace.jsonl": "e832ff2b230291d861803ae656a9d5d49ff986fd3143f0ec2d7c447a7dba76df",
        "planner_trace.json": "8d145b60a784f6bed0aa6116cd94ec0aa61ea76729a6e1cd4f52013b55a4221a",
        "run.json": "38927eae327a9fbdd42d95604496c2ca2d0535e765231196da3cf3a56ba37023",
        "effective_config.json": "58ef867d5f9f74131a2f65e37e529c8b7c53a006679fe5072c898a58af86c7fe",
        "case.json": "557dd7c5385c8bce691c07632d4577e2be68de93131752b6df0e0b94695e925a",
    },
    "v5_6_nfe1_k4__start_1787395928__seed903101": {
        "sim_trace.npz": "fcd7bcd1a4cf6e5beef744f2952fc54e5435516eddcceb8eab503e1b1dd8a95c",
        "execution_trace.jsonl": "de421fc2476ca6500940fecd8f23243e68683481bfde72fa9cf71a0569c1a628",
        "planner_trace.json": "60901fe7e20d781ac4bc92e52d97df7859680870257f76c809b11e07d3bc13f4",
        "run.json": "3a2d189d7bcc758e772c08893995a2d64bf833b7a05cd6cdd0c900a2490b469f",
        "effective_config.json": "58ef867d5f9f74131a2f65e37e529c8b7c53a006679fe5072c898a58af86c7fe",
        "case.json": "a99a9102f5b8399c4037fbd4a40f4b5d33c667a73406dd9bfbdf3c4c0bb8efb9",
    },
    "v5_6_nfe1_k4__start_1787395928__seed903102": {
        "sim_trace.npz": "df8f87b4138bd310b307d820f0bc8f259be354834087e8550e17f36e167f95e9",
        "execution_trace.jsonl": "4b6dd467934b5ec77097166eff2ccf96921f6406f3261ec470876374f1b7da12",
        "planner_trace.json": "f752f6015d2b5689df1d142f9203e78e04ce0cdc47346768da5df64daa9a6df7",
        "run.json": "a217b4d88e04831a75c71d96bd414c920b3ccc3d30b974d39517d694f9bd4fc3",
        "effective_config.json": "58ef867d5f9f74131a2f65e37e529c8b7c53a006679fe5072c898a58af86c7fe",
        "case.json": "d11394599158dc3d856c21f32f40f0ec355cb076209b7cc4a0cf6a04eec62efd",
    },
    "v5_6_nfe1_k4__start_1787395963__seed903101": {
        "sim_trace.npz": "9b859af06adc793bb9ce4819a07bc0ac0aaa3e10ee15571b10e77dcdeb5989a5",
        "execution_trace.jsonl": "60dcc5c76bd09bee27b1cdc45e47af373f49da5f59517e541473622df1f58ff8",
        "planner_trace.json": "500b0a1d0d9306b9b066d6326c7ae1d0aaa02e6a7b2c409dd03d84cadb708013",
        "run.json": "91f8959b96212ad5c007484e9bad6b8e84c64d020b535dfab979a5e8a2dc6eb0",
        "effective_config.json": "58ef867d5f9f74131a2f65e37e529c8b7c53a006679fe5072c898a58af86c7fe",
        "case.json": "45ccd4f781b503d7c95ec40a9747d142af215ab617e01be12e22c62da2807cbe",
    },
    "v5_6_nfe1_k4__start_1787395963__seed903102": {
        "sim_trace.npz": "ff1e1996cce892b87bde462f81dcae71566fed1607ecad231050be618363873a",
        "execution_trace.jsonl": "4a8c5963567d113105d2e43818124a4dfe1b4f88b22bada9ac73e7dd4a4a198f",
        "planner_trace.json": "61e07336d7e3d4328c0cf55a9bddecfd4409ee1ec5f8c271eeef23dbe5e6de17",
        "run.json": "dc85ac4c19e8f7284713d472968e0e84c8a562cfb24ab407c41a05af50c424de",
        "effective_config.json": "58ef867d5f9f74131a2f65e37e529c8b7c53a006679fe5072c898a58af86c7fe",
        "case.json": "09a4eb408c7664996d019d6962a8c27b8ebe545892aee42d20dfd1e44efd555a",
    },
    "v5_6_nfe1_k4__start_1787396028__seed903101": {
        "sim_trace.npz": "642b327766ec9b56ed8864a2298d608e5c60ead0cff06cb964fec7ff05dfd59c",
        "execution_trace.jsonl": "dd5111178b7479a0f8a143f2176ce95a9d90f2aef2c24ca812817ede550cb787",
        "planner_trace.json": "b669379480cbedc9e0cac81a319c9afad4a110de5b4fa17bc37db38a1c7cac54",
        "run.json": "3d90159092acf6af2a14d8041961531b2c9634f2f834c7988effa9ed96c4cdbc",
        "effective_config.json": "58ef867d5f9f74131a2f65e37e529c8b7c53a006679fe5072c898a58af86c7fe",
        "case.json": "7a7c73326eb097ba309f81240803492d8760f384a00397ff1c044f5afeb9c534",
    },
    "v5_6_nfe1_k4__start_1787396028__seed903102": {
        "sim_trace.npz": "87ac69f5170783489ee07ad34d283e59289f562fa988863c5a0b61ee64fb031d",
        "execution_trace.jsonl": "b8cdc7ce2d6644cd5e0055f863ac49a81cd05fcd10ad6ee5b683b3fed45ba7a6",
        "planner_trace.json": "341e6f0e82fedd2daa5fe3efd9baef7387b7159c0a7d2a87d2aef0702f5f3c37",
        "run.json": "368e91ded33ea73e7a86d444ba6b23216de7809772849fb380ca28ce5dea1653",
        "effective_config.json": "58ef867d5f9f74131a2f65e37e529c8b7c53a006679fe5072c898a58af86c7fe",
        "case.json": "4822b40456c3b5b7a8be24d3ea5dd6760c789a48c005a5aa72c24ed7494a1880",
    },
    "v5_6_nfe1_k4__start_1787396273__seed903101": {
        "sim_trace.npz": "690533608a5bdfe9e1ce70b429333dd7343273914f5b3cae63be075fb4c6382c",
        "execution_trace.jsonl": "45149781082aab9bd27a0676a92ef0ecd97d0a3e2f527635d90690eff8da0884",
        "planner_trace.json": "d214f5c287f92937db3adb68e1bab34ad515f4b204aa79ab52717a680513a13b",
        "run.json": "7f3a5e14adb136353d698668e5204b862df3f7811b6e11f4bbfb0dd5fc0f7bac",
        "effective_config.json": "58ef867d5f9f74131a2f65e37e529c8b7c53a006679fe5072c898a58af86c7fe",
        "case.json": "342b7f43b057e65ef63a96514581d5348547a4a5830568c72c7b454e3523d91b",
    },
    "v5_6_nfe1_k4__start_1787396273__seed903102": {
        "sim_trace.npz": "80935fce21dcf8cdc0bbb79d97fe3698a769a6a23745159ecf069cb4a766f3e6",
        "execution_trace.jsonl": "890c4b5275a481edd771e28cb615158813bdd4a8776b7b4a8d2c69197532c610",
        "planner_trace.json": "c72c65b3fe0d9692a10ac3db1846f376f04bf1f6308de6a73e7de475bb13db84",
        "run.json": "2de38f68686482260a88d6f4afbb7cd8de0764048f8da9f40c544f89d7faf320",
        "effective_config.json": "58ef867d5f9f74131a2f65e37e529c8b7c53a006679fe5072c898a58af86c7fe",
        "case.json": "0e52c379becbb16c2eb5e3b88ac3dda79a1bcd031bfb5c48698fe50edcedf8d1",
    },
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    assert (ROOT / "study_complete.json").is_file()
    confirm_path = ROOT / "teacher_v2_delivery_confirmation.json"
    confirm = read(confirm_path)
    assert (
        digest(confirm_path)
        == CONFIRM_SHA
        == digest(ROOT / "confirmation/campaign_snapshot.json")
    )
    selection_path = ROOT / "confirmation/selection/selection.json"
    selection = read(selection_path)
    screen_path = ROOT / "screen/selection/selection.json"
    screen = read(screen_path)
    assert digest(screen_path) == SCREEN_SELECTION_SHA
    for stage, report, count in (
        ("screen", screen, 32),
        ("confirmation", selection, 24),
    ):
        assert report["status"] == "complete_valid_matched_stage"
        assert (
            report["expected_trials"]
            == report["available_trials"]
            == len(report["trials"])
            == count
        )
        assert not report["missing_trial_keys"] and not report["invalid_trial_keys"]
        assert all(r["valid_for_selection"] for r in report["trials"])
        assert read(ROOT / stage / "progress.json")["status"] == "all_trials_completed"
    assert selection["campaign_sha256"] == CONFIRM_SHA
    expected = {
        (p["id"], c["id"], seed)
        for p in confirm["policies"]
        for c in confirm["conditions"]
        for seed in confirm["sampling_seeds"]
    }
    assert {
        (r["policy_id"], r["condition_id"], r["sampling_seed"])
        for r in selection["trials"]
    } == expected
    unchanged_screen_files = 0
    for case_name, files in PRIOR_SCREEN_RAW_HASHES.items():
        for name, expected_hash in files.items():
            assert digest(ROOT / "screen/rollouts" / case_name / name) == expected_hash
            unchanged_screen_files += 1
    recomputed_rows, case_audits = [], []
    fields = (
        "valid_for_selection",
        "clean_place",
        "strict_full_place",
        "acquired",
        "lifted",
        "carried",
        "released_in_bin",
        "dropped",
        "safety_stop",
        "actual_stop",
        "stop_reason",
        "stop_time_s",
        "wrench_limit_stop",
        "tactile_force_limit_stop",
        "completed_reason",
        "completion_time_s",
        "event_times_s",
        "final_inside_bin",
        "final_contacts_unloaded",
        "final_bin_supported_robot_unloaded",
        "final_support_forces_n",
    )
    for saved in selection["trials"]:
        folder = Path(saved["directory"])
        assert folder.is_relative_to(ROOT / "confirmation/rollouts")
        assert read(folder / "run_status.json")["status"] == "completed"
        policy = next(p for p in confirm["policies"] if p["id"] == saved["policy_id"])
        condition = next(
            c for c in confirm["conditions"] if c["id"] == saved["condition_id"]
        )
        case, run, config, info, server = map(
            read,
            (
                folder / f
                for f in (
                    "case.json",
                    "run.json",
                    "effective_config.json",
                    "policy_info.json",
                    "server_ready.json",
                )
            ),
        )
        assert case["campaign_sha256"] == CONFIRM_SHA
        assert case["checkpoint_sha256"] == policy["checkpoint_sha256"]
        assert case["server_ready_sha256"] == digest(folder / "server_ready.json")
        assert not scene_mismatches(condition_scene(confirm, condition), config)
        execution, plans = (
            read_rows(folder / "execution_trace.jsonl"),
            read_rows(folder / "planner_trace.json"),
        )
        with np.load(folder / "sim_trace.npz", allow_pickle=False) as trace:
            metrics = evaluate_policy_trace(
                trace,
                config,
                confirm["thresholds"],
                run=run,
                execution_trace=execution,
                planner_trace=plans,
            )
            errors = runtime_audit(
                confirm,
                policy,
                condition,
                info,
                server,
                run,
                trace["t"],
                metrics["control"]["stop_reason"],
            )
            assert metrics["valid_for_scoring"] and not errors
            record = {
                "policy_id": saved["policy_id"],
                "condition_id": saved["condition_id"],
                "sampling_seed": saved["sampling_seed"],
                "directory": str(folder),
                "status": "scored",
                "metrics": metrics,
            }
            row = trial_evidence(
                record, run, trace, execution, plans, confirm["thresholds"]
            )
        mismatches = [key for key in fields if row.get(key) != saved.get(key)]
        assert not mismatches, (folder.name, mismatches)
        recomputed_rows.append(row)
        case_audits.append(
            {
                "case_id": folder.name,
                "recomputed_fields_match": True,
                "runtime_valid": True,
                "sha256": {
                    f: digest(folder / f)
                    for f in (
                        "sim_trace.npz",
                        "execution_trace.jsonl",
                        "planner_trace.json",
                        "run.json",
                        "effective_config.json",
                        "case.json",
                    )
                },
            }
        )
    ranked = evaluate_selection(confirm, recomputed_rows, "confirmation")
    for key in (
        "ranking",
        "comparison",
        "confirmation_gates",
        "clear_simulator_winner",
        "conclusion",
    ):
        assert ranked[key] == selection[key], key
    a, b = [r["policy_id"] for r in ranked["ranking"]]
    by_key = {
        (r["policy_id"], r["condition_id"], r["sampling_seed"]): r
        for r in recomputed_rows
    }
    matrices = [
        np.array(
            [
                [
                    int(by_key[p, c["id"], s]["clean_place"])
                    for s in confirm["sampling_seeds"]
                ]
                for c in confirm["conditions"]
            ]
        )
        for p in (a, b)
    ]
    # Separate calculation: bootstrap paired start-level means, preserving both seed columns.
    differences = np.mean(matrices[0] - matrices[1], axis=1)
    sampled = np.random.default_rng(20260906).integers(0, 6, size=(10000, 6))
    draws = np.mean(differences[sampled], axis=1)
    limits = np.quantile(draws, [(1 - 0.95) / 2, 1 - (1 - 0.95) / 2])
    estimate = selection["comparison"]["estimate"]
    assert estimate["difference_a_minus_b"] == float(differences.mean())
    assert estimate["interval"] == limits.tolist()
    assert (
        estimate["configuration_clusters"] == 6 and estimate["seeds_per_cluster"] == 2
    )
    assert (
        estimate["bootstrap_replicates"] == 10000
        and estimate["bootstrap_seed"] == 20260906
    )
    top, other = ranked["ranking"]
    independent_gates = {
        "clean_place_at_least_8_of_12": top["clean_place"] >= 8,
        "paired_95pct_lower_bound_positive": limits[0] > 0,
        "no_increase_in_drops": top["dropped"] <= other["dropped"],
        "no_increase_in_wrench_limit_stops": top["wrench_limit_stop"]
        <= other["wrench_limit_stop"],
        "no_increase_in_tactile_force_limit_stops": top["tactile_force_limit_stop"]
        <= other["tactile_force_limit_stop"],
    }
    independent_gates = {k: bool(v) for k, v in independent_gates.items()}
    assert independent_gates == selection["confirmation_gates"]
    winner = a if all(independent_gates.values()) else None
    assert winner == selection["clear_simulator_winner"]
    with (ROOT / "confirmation/selection/trials.csv").open() as f:
        csv_rows = list(csv.DictReader(f))
    assert len(csv_rows) == 24
    for row in csv_rows:
        match = by_key[row["policy_id"], row["condition_id"], int(row["sampling_seed"])]
        for key in (
            "acquired",
            "lifted",
            "carried",
            "released_in_bin",
            "strict_full_place",
            "clean_place",
            "dropped",
            "safety_stop",
            "actual_stop",
            "wrench_limit_stop",
            "tactile_force_limit_stop",
        ):
            assert row[key] in ("True", "False")
            assert (row[key] == "True") == match[key]
    all_rows = screen["trials"] + recomputed_rows
    totals = {
        key: sum(r[key] is True for r in all_rows)
        for key in (
            "acquired",
            "lifted",
            "carried",
            "released_in_bin",
            "strict_full_place",
            "clean_place",
            "dropped",
            "actual_stop",
            "safety_stop",
            "wrench_limit_stop",
            "tactile_force_limit_stop",
        )
    }
    terminal_kinds = Counter(e for r in all_rows for e in r["terminal_safety_events"])
    out = {
        "status": "passed",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "All 24 confirmation raw metrics recomputed; previous 32 raw-score audit retained with 192 source trace/log/config hashes unchanged; all 56 valid matched denominators and final paired gates checked.",
        "screen_cases_previously_recomputed": 32,
        "screen_raw_files_unchanged": unchanged_screen_files,
        "confirmation_cases_recomputed": 24,
        "total_valid_trials": len(all_rows),
        "missing_trials": 0,
        "invalid_trials": 0,
        "hashes": {
            "prior_screen_audit": PRIOR_SCREEN_AUDIT_SHA,
            "screen_selection": digest(screen_path),
            "confirmation_selection": digest(selection_path),
            "confirmation_design": CONFIRM_SHA,
            "confirmation_csv": digest(ROOT / "confirmation/selection/trials.csv"),
            "confirmation_report": digest(ROOT / "confirmation/selection/report.md"),
            "study_complete": digest(ROOT / "study_complete.json"),
        },
        "confirmation_ranking": selection["ranking"],
        "confirmation_conclusion": selection["conclusion"],
        "clear_simulator_winner": winner,
        "paired_clean_placement": {
            "policy_a": a,
            "policy_b": b,
            "condition_ids": [c["id"] for c in confirm["conditions"]],
            "sampling_seeds": confirm["sampling_seeds"],
            "matrices": [m.tolist() for m in matrices],
            "estimate": estimate,
            "independent_gates": independent_gates,
            "bootstrap_implementation": "Independent NumPy reconstruction, resampling 6 matched start clusters with both seeds intact, 10000 replicates, seed 20260906.",
        },
        "full56_outcomes": totals,
        "full56_controller_stop_reasons": dict(
            Counter(r["stop_reason"] or "horizon_no_stop" for r in all_rows)
        ),
        "events_on_first_stopped_tick_including_nonterminal": dict(terminal_kinds),
        "case_audits": case_audits,
        "limitations": [
            "Clean-placement rate is 0/12 for both selected candidates; zero paired difference and [0,0] bootstrap interval do not establish equivalence or a hardware success probability.",
            "Each candidate has one sustained carry, both stopping at wrist extension while holding; these are pickup candidates, not completed placements.",
            "tactile_force_limit_stop includes tactile_fz or tactile_depth, labeled tactile force/depth stops.",
            "Discovery results selected candidates; confirmation contains six reserved starts and fresh seeds, with no claim of training-unseen status.",
            "Idealized contact/sensor transfer, uncertain geometry and nonisolated compute/native latency limit real-world interpretation.",
        ],
    }
    print(json.dumps(out, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
