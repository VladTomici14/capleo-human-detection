import emoji


class Messages:
    def __init__(self):
        self.font_colors = FontColors()
        self.emojis = Emojis()

    def error(self, message):
        print(f"{self.emojis.eyes}{self.font_colors.FAIL} ERROR: {message} {self.font_colors.END}")

    def failed(self, message):
        print(f"{self.emojis.cross_mark}{self.font_colors.FAIL} FAILED: {message} {self.font_colors.END}")

    def success(self, message):
        print(f"{self.emojis.sunglasses_face}{self.font_colors.OK_GREEN} SUCCESS - {message} {self.font_colors.END}")

    def warning(self, message):
        print(f"{self.emojis.warning}{self.font_colors.WARNING} WARNING: {message} {self.font_colors.END} ")


class FontColors:
    HEADER = '\033[95m'
    OK_BLUE = '\033[94m'
    OK_CYAN = '\033[96m'
    OK_GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class Emojis:
    sunglasses_face = emoji.emojize(":smiling_face_with_sunglasses:")  # 😎
    cross_mark = emoji.emojize(":cross_mark:")  # ❌
    check_mark_button = emoji.emojize(":check_mark_button:")  # ✅
    warning = emoji.emojize(":warning:")  # ⚠️
    eyes = emoji.emojize(":eyes:")  # 👀
