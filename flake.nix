{
  description = "visual-pattern-recognition: dev shell for the step 1 model test";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "aarch64-darwin" "x86_64-darwin" "x86_64-linux" "aarch64-linux" ];
      forAll = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in {
      devShells = forAll (pkgs: {
        default = pkgs.mkShell {
          # uv runs the scripts with their inline dependencies (PEP 723), so no pip installs.
          packages = [ pkgs.python313 pkgs.uv pkgs.ruff ];
        };
      });
    };
}
