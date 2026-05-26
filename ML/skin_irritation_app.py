import base64
import io
import os

import flet as ft
import joblib
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_logreg.joblib")

LABEL_TEXT = {0: "Non-irritant", 1: "Irritant"}
LABEL_KO = {0: "자극 없음", 1: "자극 있음"}

EXAMPLE_MOLS = [
    ("Acetaminophen", "CC(=O)Nc1ccc(O)cc1"),
    ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
    ("Caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
    ("Benzene", "c1ccccc1"),
    ("Ethanol", "CCO"),
]


loaded = joblib.load(MODEL_PATH)
FEATURES = loaded["features"]
SCALER = loaded["scaler"]
MODEL = loaded["model"]


def smiles_to_png_data_uri(smi: str, size: int = 320) -> str | None:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    img = Draw.MolToImage(mol, size=(size, size))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def predict(smi: str):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    desc = Descriptors.CalcMolDescriptors(mol)
    row = pd.DataFrame([desc])
    missing = [c for c in FEATURES if c not in row.columns]
    if missing:
        for c in missing:
            row[c] = 0.0
    new_X = row[FEATURES].fillna(0.0)
    new_X_scaled = SCALER.transform(new_X)
    pred = int(MODEL.predict(new_X_scaled)[0])
    prob = MODEL.predict_proba(new_X_scaled)[0]
    return {
        "pred": pred,
        "prob_non": float(prob[0]),
        "prob_irr": float(prob[1]),
        "descriptors": {f: float(new_X.iloc[0][f]) for f in FEATURES},
    }


def main(page: ft.Page):
    page.title = "Skin Irritation Predictor"
    page.width = 980
    page.height = 760
    page.padding = 24
    page.bgcolor = ft.Colors.GREY_100
    page.theme_mode = ft.ThemeMode.LIGHT
    page.scroll = ft.ScrollMode.AUTO

    header = ft.Container(
        content=ft.Column(
            spacing=4,
            controls=[
                ft.Text(
                    "Skin Irritation Predictor",
                    size=28,
                    weight=ft.FontWeight.BOLD,
                    color=ft.Colors.BLUE_GREY_900,
                ),
                ft.Text(
                    "SMILES를 입력하면 Logistic Regression 모델이 피부 자극 여부를 예측합니다.",
                    size=13,
                    color=ft.Colors.BLUE_GREY_500,
                ),
            ],
        ),
        margin=ft.Margin.only(bottom=16),
    )

    smi_input = ft.TextField(
        label="SMILES",
        hint_text="예: CC(=O)Nc1ccc(O)cc1",
        prefix_icon=ft.Icons.SCIENCE_OUTLINED,
        border_radius=12,
        filled=True,
        bgcolor=ft.Colors.WHITE,
        expand=True,
        on_submit=lambda e: on_predict(e),
    )

    predict_btn = ft.FilledButton(
        content=ft.Text("예측하기", size=14, weight=ft.FontWeight.W_600),
        icon=ft.Icons.AUTO_GRAPH,
        height=50,
        style=ft.ButtonStyle(
            shape=ft.RoundedRectangleBorder(radius=12),
            padding=ft.Padding.symmetric(horizontal=24),
        ),
        on_click=lambda e: on_predict(e),
    )

    clear_btn = ft.OutlinedButton(
        content=ft.Text("초기화", size=14),
        icon=ft.Icons.REFRESH,
        height=50,
        style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=12)),
        on_click=lambda e: reset(),
    )

    example_chips = ft.Row(
        wrap=True,
        spacing=8,
        controls=[
            ft.Chip(
                label=ft.Text(name),
                bgcolor=ft.Colors.BLUE_50,
                on_click=lambda e, s=smi: load_example(s),
            )
            for name, smi in EXAMPLE_MOLS
        ],
    )

    input_card = ft.Container(
        bgcolor=ft.Colors.WHITE,
        border_radius=16,
        padding=20,
        shadow=ft.BoxShadow(
            spread_radius=0,
            blur_radius=12,
            color=ft.Colors.with_opacity(0.06, ft.Colors.BLACK),
            offset=ft.Offset(0, 2),
        ),
        content=ft.Column(
            spacing=14,
            controls=[
                ft.Text("입력", size=14, weight=ft.FontWeight.W_600, color=ft.Colors.BLUE_GREY_700),
                ft.Row(spacing=10, controls=[smi_input]),
                ft.Row(spacing=10, controls=[predict_btn, clear_btn]),
                ft.Divider(height=1, color=ft.Colors.GREY_200),
                ft.Text("빠른 예시", size=12, color=ft.Colors.BLUE_GREY_500),
                example_chips,
            ],
        ),
    )

    mol_image = ft.Image(
        src="",
        width=320,
        height=320,
        fit=ft.BoxFit.CONTAIN,
        border_radius=12,
    )
    mol_image_placeholder = ft.Container(
        width=320,
        height=320,
        bgcolor=ft.Colors.GREY_50,
        border_radius=12,
        alignment=ft.Alignment.CENTER,
        content=ft.Column(
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            controls=[
                ft.Icon(ft.Icons.IMAGE_OUTLINED, size=48, color=ft.Colors.GREY_400),
                ft.Text("분자 구조", size=13, color=ft.Colors.GREY_500),
            ],
        ),
    )
    mol_container = ft.Container(content=mol_image_placeholder)

    label_badge = ft.Container(
        padding=ft.Padding.symmetric(horizontal=14, vertical=6),
        border_radius=20,
        bgcolor=ft.Colors.GREY_200,
        content=ft.Text("대기 중", size=12, weight=ft.FontWeight.W_600, color=ft.Colors.GREY_700),
    )
    pred_label = ft.Text("—", size=26, weight=ft.FontWeight.BOLD, color=ft.Colors.BLUE_GREY_900)
    pred_sub = ft.Text("SMILES를 입력하고 예측해 보세요", size=12, color=ft.Colors.BLUE_GREY_500)

    irr_pct = ft.Text("0.0%", size=13, weight=ft.FontWeight.W_600, color=ft.Colors.RED_400)
    non_pct = ft.Text("0.0%", size=13, weight=ft.FontWeight.W_600, color=ft.Colors.GREEN_600)
    irr_bar = ft.ProgressBar(
        value=0, color=ft.Colors.RED_400, bgcolor=ft.Colors.RED_50, border_radius=6
    )
    non_bar = ft.ProgressBar(
        value=0, color=ft.Colors.GREEN_500, bgcolor=ft.Colors.GREEN_50, border_radius=6
    )

    def prob_row(name_text, pct_text, bar):
        return ft.Column(
            spacing=4,
            controls=[
                ft.Row(
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    controls=[
                        ft.Text(name_text, size=12, color=ft.Colors.BLUE_GREY_600),
                        pct_text,
                    ],
                ),
                bar,
            ],
        )

    result_card = ft.Container(
        bgcolor=ft.Colors.WHITE,
        border_radius=16,
        padding=20,
        shadow=ft.BoxShadow(
            spread_radius=0,
            blur_radius=12,
            color=ft.Colors.with_opacity(0.06, ft.Colors.BLACK),
            offset=ft.Offset(0, 2),
        ),
        content=ft.Column(
            spacing=14,
            controls=[
                ft.Row(
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    controls=[
                        ft.Text("예측 결과", size=14, weight=ft.FontWeight.W_600, color=ft.Colors.BLUE_GREY_700),
                        label_badge,
                    ],
                ),
                pred_label,
                pred_sub,
                ft.Divider(height=1, color=ft.Colors.GREY_200),
                prob_row("자극 있음 (Irritant)", irr_pct, irr_bar),
                prob_row("자극 없음 (Non-irritant)", non_pct, non_bar),
            ],
        ),
    )

    desc_table = ft.DataTable(
        heading_row_color=ft.Colors.BLUE_GREY_50,
        heading_row_height=36,
        data_row_min_height=32,
        data_row_max_height=36,
        column_spacing=24,
        columns=[
            ft.DataColumn(ft.Text("Descriptor", size=12, weight=ft.FontWeight.W_600)),
            ft.DataColumn(ft.Text("Value", size=12, weight=ft.FontWeight.W_600), numeric=True),
        ],
        rows=[],
    )
    desc_card = ft.Container(
        bgcolor=ft.Colors.WHITE,
        border_radius=16,
        padding=20,
        shadow=ft.BoxShadow(
            spread_radius=0,
            blur_radius=12,
            color=ft.Colors.with_opacity(0.06, ft.Colors.BLACK),
            offset=ft.Offset(0, 2),
        ),
        content=ft.Column(
            spacing=10,
            controls=[
                ft.Text(
                    "사용된 Descriptor 값",
                    size=14,
                    weight=ft.FontWeight.W_600,
                    color=ft.Colors.BLUE_GREY_700,
                ),
                ft.Text(
                    f"학습 시 SelectKBest로 고른 {len(FEATURES)}개의 feature",
                    size=12,
                    color=ft.Colors.BLUE_GREY_500,
                ),
                desc_table,
            ],
        ),
    )

    image_card = ft.Container(
        bgcolor=ft.Colors.WHITE,
        border_radius=16,
        padding=20,
        shadow=ft.BoxShadow(
            spread_radius=0,
            blur_radius=12,
            color=ft.Colors.with_opacity(0.06, ft.Colors.BLACK),
            offset=ft.Offset(0, 2),
        ),
        content=ft.Column(
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=10,
            controls=[
                ft.Text(
                    "분자 구조",
                    size=14,
                    weight=ft.FontWeight.W_600,
                    color=ft.Colors.BLUE_GREY_700,
                ),
                mol_container,
            ],
        ),
    )

    def set_badge(text, fg, bg):
        label_badge.bgcolor = bg
        label_badge.content = ft.Text(text, size=12, weight=ft.FontWeight.W_600, color=fg)

    def reset():
        smi_input.value = ""
        smi_input.error_text = None
        mol_container.content = mol_image_placeholder
        pred_label.value = "—"
        pred_label.color = ft.Colors.BLUE_GREY_900
        pred_sub.value = "SMILES를 입력하고 예측해 보세요"
        set_badge("대기 중", ft.Colors.GREY_700, ft.Colors.GREY_200)
        irr_pct.value = "0.0%"
        non_pct.value = "0.0%"
        irr_bar.value = 0
        non_bar.value = 0
        desc_table.rows = []
        page.update()

    def load_example(smi: str):
        smi_input.value = smi
        page.update()
        on_predict(None)

    def on_predict(e):
        smi = (smi_input.value or "").strip()
        smi_input.error_text = None
        if not smi:
            smi_input.error_text = "SMILES를 입력하세요"
            page.update()
            return

        result = predict(smi)
        if result is None:
            smi_input.error_text = "유효하지 않은 SMILES입니다"
            mol_container.content = mol_image_placeholder
            pred_label.value = "—"
            pred_sub.value = "분자를 파싱할 수 없습니다"
            set_badge("오류", ft.Colors.WHITE, ft.Colors.RED_400)
            page.update()
            return

        data_uri = smiles_to_png_data_uri(smi)
        if data_uri:
            mol_image.src = data_uri
            mol_container.content = mol_image

        pred = result["pred"]
        p_irr = result["prob_irr"]
        p_non = result["prob_non"]

        if pred == 1:
            pred_label.value = LABEL_KO[1]
            pred_label.color = ft.Colors.RED_500
            pred_sub.value = f"이 분자는 피부 자극성을 가질 가능성이 높습니다 ({p_irr*100:.1f}%)"
            set_badge("IRRITANT", ft.Colors.WHITE, ft.Colors.RED_400)
        else:
            pred_label.value = LABEL_KO[0]
            pred_label.color = ft.Colors.GREEN_700
            pred_sub.value = f"이 분자는 피부 자극성이 낮을 것으로 예측됩니다 ({p_non*100:.1f}%)"
            set_badge("NON-IRRITANT", ft.Colors.WHITE, ft.Colors.GREEN_500)

        irr_pct.value = f"{p_irr*100:.1f}%"
        non_pct.value = f"{p_non*100:.1f}%"
        irr_bar.value = p_irr
        non_bar.value = p_non

        desc_table.rows = [
            ft.DataRow(
                cells=[
                    ft.DataCell(ft.Text(name, size=12)),
                    ft.DataCell(ft.Text(f"{value:.4f}", size=12)),
                ]
            )
            for name, value in result["descriptors"].items()
        ]

        page.update()

    body = ft.Column(
        spacing=16,
        controls=[
            input_card,
            ft.ResponsiveRow(
                spacing=16,
                run_spacing=16,
                controls=[
                    ft.Container(image_card, col={"sm": 12, "md": 5}),
                    ft.Container(result_card, col={"sm": 12, "md": 7}),
                ],
            ),
            desc_card,
        ],
    )

    page.add(header, body)


ft.run(main)
