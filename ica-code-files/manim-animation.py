from manim import *
import numpy as np
from scipy import signal

class ICAFullAnimation(Scene):
    def construct(self):
        self.camera.background_color = "#0f172a"

        self.intro_title()
        self.mixing_model_section()
        self.centering_section()
        self.whitening_section()
        self.nongaussian_section()
        self.system_conditions_section()
        self.self_signal_demo()

    # ---------------------------------------------------
    # 1. INTRO
    # ---------------------------------------------------
    def intro_title(self):
        title = Text("Independent Component Analysis (ICA)", font_size=54)
        subtitle = Text("From Mathematical Theory to Signal Separation", font_size=30)
        subtitle.next_to(title, DOWN)

        self.play(FadeIn(title, shift=UP), FadeIn(subtitle, shift=DOWN))
        self.wait(2)
        self.play(FadeOut(title), FadeOut(subtitle))

    # ---------------------------------------------------
    # 2. MIXING MODEL
    # ---------------------------------------------------
    def mixing_model_section(self):
        title = Text("1. Linear Mixing Model", font_size=42).to_edge(UP)

        eq1 = MathTex(r"\mathbf{x}(t) = A \mathbf{s}(t)")
        eq2 = MathTex(r"\mathbf{x}(t) = \begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix},\quad",
                      r"\mathbf{s}(t) = \begin{bmatrix} s_1 \\ s_2 \\ s_3 \end{bmatrix}")
        eq3 = MathTex(r"A = \text{Unknown Mixing Matrix}")

        VGroup(eq1, eq2, eq3).arrange(DOWN, buff=0.6)

        self.play(Write(title))
        self.play(Write(eq1))
        self.play(Write(eq2))
        self.play(Write(eq3))
        self.wait(3)
        self.play(FadeOut(VGroup(title, eq1, eq2, eq3)))

    # ---------------------------------------------------
    # 3. CENTERING
    # ---------------------------------------------------
    def centering_section(self):
        title = Text("2. Centering the Data", font_size=42).to_edge(UP)

        eq1 = MathTex(r"\mathbf{x}_{centered} = \mathbf{x} - E[\mathbf{x}]")
        eq2 = MathTex(r"E[\mathbf{x}_{centered}] = 0")

        VGroup(eq1, eq2).arrange(DOWN, buff=0.6)

        self.play(Write(title))
        self.play(Write(eq1))
        self.play(Write(eq2))
        self.wait(2)

        proof = Text("Centering ensures zero mean signals", font_size=28)
        proof.next_to(eq2, DOWN)
        self.play(FadeIn(proof))
        self.wait(2)

        self.play(FadeOut(VGroup(title, eq1, eq2, proof)))

    # ---------------------------------------------------
    # 4. WHITENING + PROOF
    # ---------------------------------------------------
    def whitening_section(self):
        title = Text("3. Whitening Transformation", font_size=42).to_edge(UP)

        eq1 = MathTex(r"\mathbf{z} = V D^{-1/2} V^T \mathbf{x}")
        eq2 = MathTex(r"E[\mathbf{z}\mathbf{z}^T] = I")

        proof1 = MathTex(r"E[\mathbf{x}\mathbf{x}^T] = V D V^T")
        proof2 = MathTex(r"E[\mathbf{z}\mathbf{z}^T] = V D^{-1/2} V^T (V D V^T) V D^{-1/2} V^T")
        proof3 = MathTex(r"= I")

        VGroup(eq1, eq2, proof1, proof2, proof3).arrange(DOWN, buff=0.5)

        self.play(Write(title))
        self.play(Write(eq1))
        self.play(Write(eq2))
        self.wait(2)
        self.play(Write(proof1))
        self.play(Write(proof2))
        self.play(Write(proof3))
        self.wait(3)

        self.play(FadeOut(VGroup(title, eq1, eq2, proof1, proof2, proof3)))

    # ---------------------------------------------------
    # 5. NON-GAUSSIANITY PRINCIPLE
    # ---------------------------------------------------
    def nongaussian_section(self):
        title = Text("4. Why Non-Gaussianity Matters", font_size=42).to_edge(UP)

        text1 = Text("Gaussian variables are rotationally invariant", font_size=28)
        text2 = Text("So we cannot identify independent directions!", font_size=28)
        text3 = Text("ICA finds directions maximizing non-Gaussianity", font_size=28)

        kurt = MathTex(r"\text{Kurtosis}(y) = E[y^4] - 3(E[y^2])^2")

        VGroup(text1, text2, text3, kurt).arrange(DOWN, buff=0.5)

        self.play(Write(title))
        self.play(FadeIn(text1))
        self.play(FadeIn(text2))
        self.play(FadeIn(text3))
        self.play(Write(kurt))
        self.wait(3)

        self.play(FadeOut(VGroup(title, text1, text2, text3, kurt)))

    # ---------------------------------------------------
    # 6. SYSTEM CONDITIONS
    # ---------------------------------------------------
    def system_conditions_section(self):
        title = Text("ICA System Conditions", font_size=42).to_edge(UP)
        self.play(Write(title))

        square_title = Text("1) Square System (m = n)", font_size=30, color=BLUE).shift(UP*1.5)
        square_eq1 = MathTex(r"\mathbf{x} = A \mathbf{s}")
        square_eq2 = MathTex(r"W = A^{-1}")
        square_eq3 = MathTex(r"\hat{\mathbf{s}} = W \mathbf{x}")

        square_group = VGroup(square_eq1, square_eq2, square_eq3).arrange(DOWN, buff=0.5)
        square_group.next_to(square_title, DOWN)

        self.play(FadeIn(square_title))
        self.play(Write(square_eq1))
        self.play(Write(square_eq2))
        self.play(Write(square_eq3))
        self.wait(2)

        square_note = Text("If A is full rank, sources are recoverable up to scale & permutation", font_size=24)
        square_note.to_edge(DOWN)
        self.play(FadeIn(square_note))
        self.wait(2)

        self.play(FadeOut(VGroup(square_title, square_group, square_note)))

        over_title = Text("2) Overdetermined System (m > n)", font_size=30, color=GREEN).shift(UP*1.5)

        over_eq1 = MathTex(r"\mathbf{x} = A \mathbf{s}, \quad A \in \mathbb{R}^{m \times n},\ m>n")
        over_eq2 = MathTex(r"\text{Cov}(\mathbf{x}) = A\,\text{Cov}(\mathbf{s})\,A^T")
        over_eq3 = MathTex(r"\text{Use PCA: } \mathbf{z} = V_k^T \mathbf{x}")
        over_eq4 = MathTex(r"\text{Then apply ICA on reduced } \mathbf{z}")

        over_group = VGroup(over_eq1, over_eq2, over_eq3, over_eq4).arrange(DOWN, buff=0.5)
        over_group.next_to(over_title, DOWN)

        self.play(FadeIn(over_title))
        self.play(Write(over_eq1))
        self.play(Write(over_eq2))
        self.play(Write(over_eq3))
        self.play(Write(over_eq4))
        self.wait(3)

        over_note = Text("Extra sensors add redundancy → PCA removes noise before ICA", font_size=24)
        over_note.to_edge(DOWN)
        self.play(FadeIn(over_note))
        self.wait(2)

        self.play(FadeOut(VGroup(over_title, over_group, over_note)))

        under_title = Text("3) Underdetermined System (m < n)", font_size=30, color=RED).shift(UP*1.5)

        under_eq1 = MathTex(r"\mathbf{x} = A \mathbf{s}, \quad m<n")
        under_eq2 = MathTex(r"\text{Infinite solutions for } \mathbf{s}")
        under_eq3 = MathTex(r"\text{Assume sparsity: } \min ||\mathbf{s}||_1 \ \text{s.t. } \mathbf{x}=A\mathbf{s}")
        under_eq4 = MathTex(r"\text{Use Sparse ICA / Compressed Sensing}")

        under_group = VGroup(under_eq1, under_eq2, under_eq3, under_eq4).arrange(DOWN, buff=0.5)
        under_group.next_to(under_title, DOWN)

        self.play(FadeIn(under_title))
        self.play(Write(under_eq1))
        self.play(Write(under_eq2))
        self.play(Write(under_eq3))
        self.play(Write(under_eq4))
        self.wait(3)

        under_note = Text("We add assumptions (sparsity, independence) to make recovery possible", font_size=24)
        under_note.to_edge(DOWN)
        self.play(FadeIn(under_note))
        self.wait(3)

        self.play(FadeOut(VGroup(title, under_title, under_group, under_note)))

    # ---------------------------------------------------
    # 7. SIGNAL DEMO WITH ICA
    # ---------------------------------------------------
    def self_signal_demo(self):
        title = Text("5. ICA Signal Separation Pipeline", font_size=42).to_edge(UP)
        self.play(Write(title))

        t = np.linspace(0, 1, 400)
        s1 = np.sin(2 * np.pi * 5 * t)
        s2 = signal.square(2 * np.pi * 3 * t)
        s3 = signal.sawtooth(2 * np.pi * 2 * t)
        S = np.vstack([s1, s2, s3])

        axes_src = VGroup(*[
            Axes(x_range=[0,1,0.2], y_range=[-2,2,1], x_length=4, y_length=2)
            for _ in range(3)
        ]).arrange(DOWN, buff=0.6).to_edge(LEFT)

        self.play(Create(axes_src))

        def plot(ax, sig, color):
            return ax.plot_line_graph(t, sig, add_vertex_dots=False).set_color(color)

        g1 = plot(axes_src[0], s1, BLUE)
        g2 = plot(axes_src[1], s2, GREEN)
        g3 = plot(axes_src[2], s3, RED)

        src_label = Text("Original Independent Sources", font_size=26).next_to(axes_src, UP)
        self.play(Create(g1), Create(g2), Create(g3), FadeIn(src_label))
        self.wait(2)

        A = np.array([[1, 0.5, 0.3],
                      [0.4, 1, 0.2],
                      [0.2, 0.3, 1]])

        mix_matrix = MathTex(
            r"A = \begin{bmatrix} 1 & 0.5 & 0.3 \\ 0.4 & 1 & 0.2 \\ 0.2 & 0.3 & 1 \end{bmatrix}"
        ).scale(0.8).to_edge(RIGHT).shift(UP*2)

        self.play(Write(mix_matrix))

        X = A @ S

        axes_mix = axes_src.copy().to_edge(RIGHT)
        mix_label = Text("Mixed Signals", font_size=26).next_to(axes_mix, UP)

        self.play(TransformFromCopy(axes_src, axes_mix), FadeIn(mix_label))

        m1 = plot(axes_mix[0], X[0], YELLOW)
        m2 = plot(axes_mix[1], X[1], ORANGE)
        m3 = plot(axes_mix[2], X[2], PURPLE)

        self.play(TransformFromCopy(g1, m1),
                  TransformFromCopy(g2, m2),
                  TransformFromCopy(g3, m3))
        self.wait(2)

        self.play(FadeOut(src_label), FadeOut(g1), FadeOut(g2), FadeOut(g3), FadeOut(axes_src))

        step1 = Text("Step 1: Centering (Zero Mean)", font_size=26).to_edge(DOWN)
        self.play(FadeIn(step1))

        X_centered = X - X.mean(axis=1, keepdims=True)
        self.wait(2)
        self.play(FadeOut(step1))

        step2 = Text("Step 2: Whitening (Remove Correlation)", font_size=26).to_edge(DOWN)
        self.play(FadeIn(step2))

        cov = np.cov(X_centered)
        d, E = np.linalg.eigh(cov)
        D_inv = np.diag(1/np.sqrt(d))
        X_white = D_inv @ E.T @ X_centered

        self.wait(2)
        self.play(FadeOut(step2))

        step3 = Text("Step 3: ICA Rotation (Maximize Non-Gaussianity)", font_size=26).to_edge(DOWN)
        self.play(FadeIn(step3))

        theta = PI/4
        W = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0, 0, 1]
        ])

        S_est = W @ X_white
        self.wait(2)
        self.play(FadeOut(step3), FadeOut(mix_label), FadeOut(m1), FadeOut(m2), FadeOut(m3), FadeOut(axes_mix), FadeOut(mix_matrix))

        axes_rec = VGroup(*[
            Axes(x_range=[0,1,0.2], y_range=[-2,2,1], x_length=4, y_length=2)
            for _ in range(3)
        ]).arrange(DOWN, buff=0.6).to_edge(RIGHT)

        rec_label = Text("Recovered Sources (ICA Output)", font_size=26).next_to(axes_rec, UP)
        self.play(Create(axes_rec), FadeIn(rec_label))

        r1 = plot(axes_rec[0], S_est[0], BLUE)
        r2 = plot(axes_rec[1], S_est[1], GREEN)
        r3 = plot(axes_rec[2], S_est[2], RED)

        self.play(Create(r1), Create(r2), Create(r3))
        self.wait(2)

        axes_orig = axes_rec.copy().to_edge(LEFT)
        orig_label = Text("Original Sources", font_size=26).next_to(axes_orig, UP)

        self.play(FadeIn(orig_label), TransformFromCopy(axes_rec, axes_orig))

        g1o = plot(axes_orig[0], s1, BLUE)
        g2o = plot(axes_orig[1], s2, GREEN)
        g3o = plot(axes_orig[2], s3, RED)

        self.play(Create(g1o), Create(g2o), Create(g3o))

        arrows = VGroup(*[
            Arrow(axes_orig[i].get_right(), axes_rec[i].get_left(), buff=0.2)
            for i in range(3)
        ])

        compare_text = Text("Recovered ≈ Original (up to scale & permutation)", font_size=24).to_edge(DOWN)

        self.play(Create(arrows), FadeIn(compare_text))
        self.wait(4)

        self.play(*[FadeOut(m) for m in self.mobjects])
