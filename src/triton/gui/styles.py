APP_CSS = """
<style>
		:root {
			--bg-top: #06141f;
			--bg-mid: #0e3040;
			--bg-bottom: #d7e8ea;
			--panel: rgba(7, 30, 43, 0.74);
			--panel-border: rgba(143, 205, 206, 0.22);
			--panel-soft: rgba(250, 252, 252, 0.06);
			--accent: #ff9f1c;
			--accent-soft: #ffd6a0;
			--ink: #ebf6f7;
			--muted: #b7d0d4;
			--hero-start: rgba(4, 27, 39, 0.92);
			--hero-end: rgba(14, 61, 78, 0.88);
			--hero-kicker: #ffd6a0;
			--hero-body: #cbe2e5;
			--hero-pill-bg: rgba(255, 159, 28, 0.16);
			--hero-pill-text: #ffe9c7;
			--card-top-bg: rgba(255, 255, 255, 0.06);
			--card-subtle-text: #cbe2e5;
			--metric-start: rgba(9, 40, 55, 0.95);
			--metric-end: rgba(9, 40, 55, 0.7);
		}

		html[data-theme="light"], body[data-theme="light"], [data-theme="light"] {
			--bg-top: #f6efe4;
			--bg-mid: #f7fbff;
			--bg-bottom: #e7f1f3;
			--panel: rgba(255, 255, 255, 0.92);
			--panel-border: rgba(16, 41, 53, 0.16);
			--panel-soft: rgba(10, 37, 50, 0.05);
			--ink: #102935;
			--muted: #385767;
			--hero-start: rgba(255, 255, 255, 0.96);
			--hero-end: rgba(235, 247, 250, 0.98);
			--hero-kicker: #8f4b00;
			--hero-body: #26485a;
			--hero-pill-bg: rgba(255, 159, 28, 0.22);
			--hero-pill-text: #5f3400;
			--card-top-bg: rgba(255, 255, 255, 0.9);
			--card-subtle-text: #2e4f5f;
			--metric-start: rgba(255, 255, 255, 0.97);
			--metric-end: rgba(240, 248, 251, 0.97);
		}

		.stApp {
			background:
				radial-gradient(circle at top left, rgba(255, 159, 28, 0.16), transparent 30%),
				linear-gradient(180deg, var(--bg-top) 0%, var(--bg-mid) 45%, #153e50 72%, var(--bg-bottom) 140%);
		}

		[data-testid="stHeader"] {
			background: transparent;
		}

		[data-testid="stSidebar"] {
			background: rgba(5, 20, 33, 0.9);
			border-right: 1px solid rgba(143, 205, 206, 0.18);
		}

		html[data-theme="light"] [data-testid="stSidebar"],
		body[data-theme="light"] [data-testid="stSidebar"],
		[data-theme="light"] [data-testid="stSidebar"] {
			background: rgba(255, 255, 255, 0.94);
			border-right: 1px solid rgba(16, 41, 53, 0.12);
		}

		h1, h2, h3, h4, h5, h6, p, label, div, span {
			color: var(--ink) !important;
		}

		div[data-testid="stMetric"] {
			background: linear-gradient(180deg, var(--metric-start), var(--metric-end));
			border: 1px solid var(--panel-border);
			padding: 16px;
			border-radius: 18px;
			box-shadow: 0 18px 40px rgba(2, 10, 16, 0.22);
			transition: transform 0.2s ease, box-shadow 0.2s ease;
		}

		div[data-testid="stMetric"]:hover {
			transform: translateY(-2px);
			box-shadow: 0 22px 48px rgba(2, 10, 16, 0.28);
		}

		.stButton > button, .stDownloadButton > button, button[kind="primary"] {
			background: linear-gradient(135deg, #ff9f1c, #ffbf69) !important;
			color: #08202c !important;
			border: none !important;
			border-radius: 999px !important;
			font-weight: 700 !important;
			box-shadow: 0 12px 24px rgba(255, 159, 28, 0.28);
			transition: transform 0.15s ease, box-shadow 0.15s ease;
		}

		.stButton > button:hover, .stDownloadButton > button:hover, button[kind="primary"]:hover {
			transform: translateY(-1px);
			box-shadow: 0 16px 32px rgba(255, 159, 28, 0.36);
		}

		.stButton > button *, .stDownloadButton > button *, button[kind="primary"] * {
			color: #08202c !important;
		}

		html[data-theme="light"] .stButton > button[kind="secondary"],
		html[data-theme="light"] .stButton > button[kind="tertiary"],
		html[data-theme="light"] .stDownloadButton > button[kind="secondary"],
		html[data-theme="light"] .stDownloadButton > button[kind="tertiary"],
		body[data-theme="light"] .stButton > button[kind="secondary"],
		body[data-theme="light"] .stButton > button[kind="tertiary"],
		body[data-theme="light"] .stDownloadButton > button[kind="secondary"],
		body[data-theme="light"] .stDownloadButton > button[kind="tertiary"],
		[data-theme="light"] .stButton > button[kind="secondary"],
		[data-theme="light"] .stButton > button[kind="tertiary"],
		[data-theme="light"] .stDownloadButton > button[kind="secondary"],
		[data-theme="light"] .stDownloadButton > button[kind="tertiary"] {
			color: #102935 !important;
			border: 1px solid rgba(16, 41, 53, 0.22) !important;
			background: rgba(255, 255, 255, 0.96) !important;
		}

		html[data-theme="light"] .stButton > button[kind="secondary"] *,
		html[data-theme="light"] .stButton > button[kind="tertiary"] *,
		html[data-theme="light"] .stDownloadButton > button[kind="secondary"] *,
		html[data-theme="light"] .stDownloadButton > button[kind="tertiary"] *,
		body[data-theme="light"] .stButton > button[kind="secondary"] *,
		body[data-theme="light"] .stButton > button[kind="tertiary"] *,
		body[data-theme="light"] .stDownloadButton > button[kind="secondary"] *,
		body[data-theme="light"] .stDownloadButton > button[kind="tertiary"] *,
		[data-theme="light"] .stButton > button[kind="secondary"] *,
		[data-theme="light"] .stButton > button[kind="tertiary"] *,
		[data-theme="light"] .stDownloadButton > button[kind="secondary"] *,
		[data-theme="light"] .stDownloadButton > button[kind="tertiary"] * {
			color: #102935 !important;
		}

		@media (prefers-color-scheme: light) {
			.stButton > button[kind="secondary"],
			.stButton > button[kind="tertiary"],
			.stDownloadButton > button[kind="secondary"],
			.stDownloadButton > button[kind="tertiary"] {
				color: #102935 !important;
				border: 1px solid rgba(16, 41, 53, 0.22) !important;
				background: rgba(255, 255, 255, 0.96) !important;
			}

			.stButton > button[kind="secondary"] *,
			.stButton > button[kind="tertiary"] *,
			.stDownloadButton > button[kind="secondary"] *,
			.stDownloadButton > button[kind="tertiary"] * {
				color: #102935 !important;
			}
		}

		.stTextInput input, .stSelectbox div[data-baseweb="select"], .stTextArea textarea {
			background: rgba(255, 255, 255, 0.06) !important;
			border-radius: 14px !important;
		}

		html[data-theme="light"] .stTextInput input,
		html[data-theme="light"] .stSelectbox div[data-baseweb="select"],
		html[data-theme="light"] .stTextArea textarea,
		body[data-theme="light"] .stTextInput input,
		body[data-theme="light"] .stSelectbox div[data-baseweb="select"],
		body[data-theme="light"] .stTextArea textarea,
		[data-theme="light"] .stTextInput input,
		[data-theme="light"] .stSelectbox div[data-baseweb="select"],
		[data-theme="light"] .stTextArea textarea {
			background: rgba(255, 255, 255, 0.92) !important;
			border: 1px solid rgba(16, 41, 53, 0.18) !important;
		}

		.stFileUploader, .stAudio, .stForm {
			background: var(--panel);
			border: 1px solid var(--panel-border);
			border-radius: 20px;
			padding: 12px;
			box-shadow: 0 18px 40px rgba(2, 10, 16, 0.16);
		}

		div[data-testid="stExpander"] {
			border: 1px solid var(--panel-border);
			border-radius: 16px;
			background: var(--panel);
		}

		div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
			border-radius: 16px;
			border-color: var(--panel-border);
		}

		.stDataFrame {
			border-radius: 12px;
			overflow: hidden;
		}

		div[data-baseweb="tab-list"] {
			gap: 8px;
		}

		button[data-baseweb="tab"] {
			background: rgba(255, 255, 255, 0.06);
			border-radius: 999px;
			padding: 8px 18px;
			transition: background 0.2s ease, box-shadow 0.2s ease;
			border: 1px solid transparent;
		}

		button[data-baseweb="tab"]:hover {
			background: rgba(255, 255, 255, 0.1);
		}

		button[data-baseweb="tab"][aria-selected="true"] {
			background: rgba(255, 159, 28, 0.22);
			border: 1px solid rgba(255, 159, 28, 0.3);
			box-shadow: 0 4px 16px rgba(255, 159, 28, 0.15);
		}

		html[data-theme="light"] button[data-baseweb="tab"],
		body[data-theme="light"] button[data-baseweb="tab"],
		[data-theme="light"] button[data-baseweb="tab"] {
			background: rgba(255, 255, 255, 0.85);
			border: 1px solid rgba(16, 41, 53, 0.14);
		}

		/* Force dark theme even when user/browser selects light mode. */
		html[data-theme="light"],
		body[data-theme="light"],
		[data-theme="light"] {
			color-scheme: dark !important;
			--bg-top: #06141f !important;
			--bg-mid: #0e3040 !important;
			--bg-bottom: #d7e8ea !important;
			--panel: rgba(7, 30, 43, 0.74) !important;
			--panel-border: rgba(143, 205, 206, 0.22) !important;
			--panel-soft: rgba(250, 252, 252, 0.06) !important;
			--ink: #ebf6f7 !important;
			--muted: #b7d0d4 !important;
			--hero-start: rgba(4, 27, 39, 0.92) !important;
			--hero-end: rgba(14, 61, 78, 0.88) !important;
			--hero-kicker: #ffd6a0 !important;
			--hero-body: #cbe2e5 !important;
			--hero-pill-bg: rgba(255, 159, 28, 0.16) !important;
			--hero-pill-text: #ffe9c7 !important;
			--card-top-bg: rgba(255, 255, 255, 0.06) !important;
			--card-subtle-text: #cbe2e5 !important;
			--metric-start: rgba(9, 40, 55, 0.95) !important;
			--metric-end: rgba(9, 40, 55, 0.7) !important;
		}

		html[data-theme="light"] [data-testid="stSidebar"],
		body[data-theme="light"] [data-testid="stSidebar"],
		[data-theme="light"] [data-testid="stSidebar"] {
			background: rgba(5, 20, 33, 0.9) !important;
			border-right: 1px solid rgba(143, 205, 206, 0.18) !important;
		}

		html[data-theme="light"] .stTextInput input,
		html[data-theme="light"] .stSelectbox div[data-baseweb="select"],
		html[data-theme="light"] .stTextArea textarea,
		body[data-theme="light"] .stTextInput input,
		body[data-theme="light"] .stSelectbox div[data-baseweb="select"],
		body[data-theme="light"] .stTextArea textarea,
		[data-theme="light"] .stTextInput input,
		[data-theme="light"] .stSelectbox div[data-baseweb="select"],
		[data-theme="light"] .stTextArea textarea {
			background: rgba(255, 255, 255, 0.06) !important;
			border: 1px solid rgba(143, 205, 206, 0.22) !important;
		}

		html[data-theme="light"] .stButton > button[kind="secondary"],
		html[data-theme="light"] .stButton > button[kind="tertiary"],
		html[data-theme="light"] .stDownloadButton > button[kind="secondary"],
		html[data-theme="light"] .stDownloadButton > button[kind="tertiary"],
		body[data-theme="light"] .stButton > button[kind="secondary"],
		body[data-theme="light"] .stButton > button[kind="tertiary"],
		body[data-theme="light"] .stDownloadButton > button[kind="secondary"],
		body[data-theme="light"] .stDownloadButton > button[kind="tertiary"],
		[data-theme="light"] .stButton > button[kind="secondary"],
		[data-theme="light"] .stButton > button[kind="tertiary"],
		[data-theme="light"] .stDownloadButton > button[kind="secondary"],
		[data-theme="light"] .stDownloadButton > button[kind="tertiary"] {
			color: #ebf6f7 !important;
			border: 1px solid rgba(143, 205, 206, 0.22) !important;
			background: rgba(255, 255, 255, 0.06) !important;
		}

		html[data-theme="light"] .stButton > button[kind="secondary"] *,
		html[data-theme="light"] .stButton > button[kind="tertiary"] *,
		html[data-theme="light"] .stDownloadButton > button[kind="secondary"] *,
		html[data-theme="light"] .stDownloadButton > button[kind="tertiary"] *,
		body[data-theme="light"] .stButton > button[kind="secondary"] *,
		body[data-theme="light"] .stButton > button[kind="tertiary"] *,
		body[data-theme="light"] .stDownloadButton > button[kind="secondary"] *,
		body[data-theme="light"] .stDownloadButton > button[kind="tertiary"] *,
		[data-theme="light"] .stButton > button[kind="secondary"] *,
		[data-theme="light"] .stButton > button[kind="tertiary"] *,
		[data-theme="light"] .stDownloadButton > button[kind="secondary"] *,
		[data-theme="light"] .stDownloadButton > button[kind="tertiary"] * {
			color: #ebf6f7 !important;
		}

		html[data-theme="light"] button[data-baseweb="tab"],
		body[data-theme="light"] button[data-baseweb="tab"],
		[data-theme="light"] button[data-baseweb="tab"] {
			background: rgba(255, 255, 255, 0.06) !important;
			border: 1px solid rgba(143, 205, 206, 0.18) !important;
		}
		</style>
		"""
        